"""The engineered T0 agent: battery -> graph-build -> plan -> act.

Stage 1b of docs/DESIGN-2026-08-14-engineered-agent.md. Flow per game:

  1. Probe battery (Stage-1a `ProbeBattery`, <= 16 actions) learns the HUD
     mask + SELF/REACTIVE/DEAD + archetype. Battery steps are not wasted:
     every transition (including the battery's own) is recorded raw and
     replayed into the transition graph once the perception exists.
  2. Exact per-level `LevelGraph` over masked states; per-level model reset
     (the graph of a level persists across GAME_OVER — knowledge survives
     death — but each new level starts a fresh graph).
  3. T0 planner: cheapest-path-to-frontier with battery priors; RESET-aware;
     every planned edge is verified against the env and any mismatch aborts
     the plan and replans.
  4. Per-level re-probe on level-up (lf52 lesson), interleaved through the
     same recorder so probe actions are counted and their transitions kept.
  5. Budgets: hard per-game action budget + wall-clock; optional per-level
     stop-loss keyed to the human actions-per-completed-level medians
     (median x multiplier policy, design doc §3).

Action accounting matches the scored-action law: every env.step (including
mid-game RESETs) counts 1; the game-opening reset() is the only free call.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import ndimage

if __package__ in (None, ""):  # running as a bare script: put src/ on the path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "engineered"

from engineered.battery import (
    BatteryConfig,
    BatteryState,
    PerceptionProfile,
    ProbeBattery,
)
from engineered.effects import EffectConfig, EffectEngine
from engineered.envs import game_stem
from engineered.graph import (
    ActionKey,
    LevelGraph,
    NodeKey,
    RESET_ACTION,
    click_key,
    simple_key,
)
from engineered.perception import Perception, settled_frame
from engineered.planner import Plan, plan

_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


# ---------------------------------------------------------------------------
# Config / report
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    budget: int = 4000                 # hard per-game scored-action budget
    wall_s: float = 900.0              # hard per-game wall-clock
    battery_budget: int = 16
    reprobe_budget: int = 10
    reprobe_on_level_up: bool = True
    # tier-0 click menu cap per state. 48, not 24: ft09's mechanic is a
    # 36-tile grid (64 components on the board) and a 24-cap left the
    # winning tiles unreachable at any tier.
    max_menu_clicks: int = 48
    lattice_steps: tuple[int, ...] = (8, 4)  # tier-1, tier-2 grids
    prior_penalty: float = 0.3
    greedy_rank_max: int = 3
    # per-level stop-loss: None disables; else level abandoned (game over
    # for us — levels are sequential) past multiplier * human_apl[stem]
    level_stop_multiplier: float | None = None
    human_apl: dict[str, float] = field(default_factory=dict)
    # Stage 2a: T1/T2 effect model + predicted-edge planning
    effects_enabled: bool = True
    effect_config: EffectConfig = field(default_factory=EffectConfig)
    verbose: bool = False


@dataclass
class GameReport:
    stem: str
    game_id: str
    won: bool
    levels_completed: int
    win_levels: int | None
    actions_total: int
    per_level_actions: dict[int, int]
    battery_actions: int
    plan_mismatches: int
    violation_events: int
    violated_states: int
    end_reason: str
    wall_s: float
    archetype: str
    graph_stats: dict[int, dict[str, int]]
    # Stage 2a additions (defaulted so Stage-1 call sites stay valid)
    effect_stats: list[dict[str, Any]] = field(default_factory=list)
    rule_mispredictions: int = 0     # live predicted-edge failures
    predicted_steps_taken: int = 0   # executed steps that rode predicted edges
    # Stage 2b additions
    audit_plans: int = 0             # forced probes of rule-predicted pairs
    win_pred_attempts: int = 0       # predicted-win actions executed
    win_pred_hits: int = 0           # ... that actually leveled up / won

    @property
    def completed_level_actions(self) -> list[int]:
        return [self.per_level_actions.get(l, 0) for l in range(self.levels_completed)]

    @property
    def apl_mean(self) -> float | None:
        done = self.completed_level_actions
        return (sum(done) / len(done)) if done else None


# ---------------------------------------------------------------------------
# Recording env proxy — single choke point for actions + transitions
# ---------------------------------------------------------------------------

class RecordingEnv:
    """Wraps an engine env; the agent sees every reset/step exactly once."""

    def __init__(self, env: Any, sink: "EngineeredAgent") -> None:
        self._env = env
        self._sink = sink
        self.actions = 0
        self.last_obs: Any = None

    def reset(self) -> Any:
        obs = self._env.reset()
        self.last_obs = obs
        self._sink._on_reset(obs)
        return obs

    def step(self, action: Any, data: dict[str, int] | None = None) -> Any:
        obs = self._env.step(action, data=data) if data else self._env.step(action)
        self.actions += 1
        self.last_obs = obs
        self._sink._on_step(action, data, obs)
        return obs


def _action_id(action: Any) -> int:
    """GameAction -> integer id (RESET = 0)."""
    value = getattr(action, "value", None)
    if isinstance(value, int):
        return value
    name = getattr(action, "name", str(action))
    if "RESET" in name:
        return 0
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


# one raw transition, buffered until the perception exists
# (prev_frame, prev_avail, lc_before, akey, frame, avail_after, state, lc_after)
_RawT = tuple[np.ndarray, tuple[int, ...], int, ActionKey, np.ndarray,
              tuple[int, ...], str, int]


# ---------------------------------------------------------------------------
# Planner port over the effect engine (design doc §2 Layer 4)
# ---------------------------------------------------------------------------

class _PlannerEffects:
    """Adapts one level's graph + the game's EffectEngine to EffectsPort."""

    def __init__(self, engine: EffectEngine, graph: LevelGraph) -> None:
        self.engine = engine
        self.graph = graph
        self.max_pred_depth = engine.cfg.max_pred_depth
        self.max_pred_nodes = engine.cfg.max_pred_nodes

    @property
    def active(self) -> bool:
        return bool(self.engine.gated_rules())

    def classify(self, key: NodeKey, akey: ActionKey):
        node = self.graph.nodes.get(key)
        if node is None:
            return None
        return self.engine.predict(key, node.frame, akey)

    def chain(self, key: Any, frame: np.ndarray, avail: tuple[int, ...]):
        return self.engine.chain_successors(key, frame, avail)

    @property
    def has_win(self) -> bool:
        return self.engine.has_win_rules

    def win_candidates(self, key: NodeKey, menu: list[ActionKey]) -> list[ActionKey]:
        node = self.graph.nodes.get(key)
        if node is None:
            return []
        return self.engine.win_candidates(node.frame, node.avail, menu)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class EngineeredAgent:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.cfg = config or AgentConfig()

    # -- recorder callbacks ------------------------------------------------

    def _on_reset(self, obs: Any) -> None:
        self._prev_frame = settled_frame(obs).copy()
        self._prev_avail = tuple(sorted(a for a in (obs.available_actions or []) if a))
        self._level = int(obs.levels_completed or 0)
        self._cur_key = None

    def _on_step(self, action: Any, data: dict[str, int] | None, obs: Any) -> None:
        aid = _action_id(action)
        if aid == 6 and data:
            akey = click_key(data["x"], data["y"])
        else:
            akey = simple_key(aid)
        frame = settled_frame(obs).copy()
        avail = tuple(sorted(a for a in (obs.available_actions or []) if a))
        lc_after = int(obs.levels_completed or 0)
        state = str(obs.state)

        self.level_actions[self._level] += 1
        self.action_tries[akey] += 1
        t: _RawT = (self._prev_frame, self._prev_avail, self._level,
                    akey, frame, avail, state, lc_after)
        if self.perception is None:
            self._pending.append(t)
        else:
            self._ingest(t)
        self._prev_frame = frame
        self._prev_avail = avail
        self._level = lc_after
        self.max_level = max(self.max_level, lc_after)

    # -- graph ingestion ---------------------------------------------------

    def _graph(self, level: int) -> LevelGraph:
        g = self.graphs.get(level)
        if g is None:
            g = LevelGraph(self.stem, level)
            self.graphs[level] = g
        return g

    def _replay_pending(self) -> None:
        for t in self._pending:
            self._ingest(t)
        self._pending.clear()

    def _ingest(self, t: _RawT) -> None:
        (prev_frame, prev_avail, lc_before, akey, frame, avail,
         state, lc_after) = t
        assert self.perception is not None
        p = self.perception
        g = self._graph(lc_before)
        from arcengine import GameState

        if self._cur_key is None:
            if akey == RESET_ACTION:
                # RESET out of a death screen: the source is not a state;
                # just (re-)root at the landing frame.
                dst = g.effective_key(p.state_key(frame), None)
                g.ensure_node(dst, frame, avail)
                g.set_root(dst)
                self._cur_key = dst
                return
            src = g.effective_key(p.state_key(prev_frame), None)
            g.ensure_node(src, prev_frame, prev_avail)
            g.set_root(src)
        else:
            src = self._cur_key

        if state == str(GameState.GAME_OVER):
            g.observe(src, akey, None, game_over=True)
            self._cur_key = None
        elif state == str(GameState.WIN) or lc_after > lc_before:
            g.observe(src, akey, None, level_up=True)
            # Stage 2b: win transitions feed the cross-level win predicate
            # (they stay OUT of effects.observe — rules model board dynamics)
            if self.effects_engine is not None:
                self.effects_engine.observe_win(prev_frame, akey)
            if state == str(GameState.WIN):
                self._cur_key = None
            else:
                ng = self._graph(lc_after)
                dst = ng.effective_key(p.state_key(frame), None)
                ng.ensure_node(dst, frame, avail)
                ng.set_root(dst)
                self._cur_key = dst
        else:
            dst = g.effective_key(p.state_key(frame), (src, akey))
            g.ensure_node(dst, frame, avail)
            g.observe(src, akey, dst)
            if dst == src:
                self.action_nulls[akey] += 1
            # T1/T2 rule learning: prequential test-then-train on every
            # within-level, non-terminal transition (terminal transitions
            # model level boundaries, not board dynamics — excluded)
            if self.effects_engine is not None:
                self.effects_engine.observe(prev_frame, akey, frame)
            self._cur_key = dst

    # -- menus -------------------------------------------------------------

    def _click_targets(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """(reactive_overlap, size, y, x) per component, ranked: components
        overlapping battery-REACTIVE cells first, then larger first. Click
        point = component pixel nearest the centroid (Stage-0 trap 1)."""
        assert self.perception is not None
        board = self.perception.board_mask
        f = np.asarray(frame, dtype=np.int16)
        bg = int(np.bincount(f[board].ravel()).argmax()) if board.any() \
            else int(np.bincount(f.ravel()).argmax())
        fb = np.where(board, f, bg)
        out: list[tuple[int, int, int, int]] = []
        for col in np.unique(fb):
            if col == bg:
                continue
            lab, n = ndimage.label(fb == col, structure=_S4)
            for i in range(1, n + 1):
                ys, xs = np.nonzero(lab == i)
                cy, cx = ys.mean(), xs.mean()
                k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
                overlap = int(self.reactive_mask[ys, xs].sum()) \
                    if self.reactive_mask is not None else 0
                out.append((overlap, len(ys), int(ys[k]), int(xs[k])))
        out.sort(key=lambda r: (-r[0], -r[1], r[2], r[3]))
        return out

    def _menu(self, g: LevelGraph, key: NodeKey, tier: int) -> list[ActionKey]:
        """Ranked action menu for one node, cached per (level, key, tier).

        Two priors, both game-agnostic:
          * bucket order by archetype — the small aux verb set (ACTION5/7,
            "submit"/"undo" in several games) is 1-2 actions per state and
            goes BEFORE the wide click sweep on click games; burying it
            behind 24 clicks starves games where ACTION5 completes the level.
          * within a bucket, ascending GLOBAL try-count of the exact action
            key (count-based novelty — an action probed at hundreds of other
            states teaches less than one probed nowhere), tie-broken by the
            battery's REACTIVE overlap then component size. Counts are frozen
            at menu build time (a node's menu is built when it is first
            planned from, so counts are fresh where it matters).
        """
        engine = self.effects_engine
        cache_key = (g.level, key, tier,
                     engine.version if engine is not None else -1)
        cached = self._menu_cache.get(cache_key)
        if cached is not None:
            return cached
        node = g.nodes.get(key)
        if node is None:
            return []
        avail = node.avail
        tries = self.action_tries

        def by_count(keys: list[ActionKey]) -> list[ActionKey]:
            return sorted(keys, key=lambda a: (tries[a], keys.index(a)))

        dirs = by_count([simple_key(a) for a in (1, 2, 3, 4) if a in avail])
        aux = by_count([simple_key(a) for a in (5, 7) if a in avail])
        clicks: list[ActionKey] = []
        if 6 in avail:
            targets = self._click_targets(node.frame)[: self.cfg.max_menu_clicks]
            ranked = sorted(
                ((tries[click_key(x, y)], -ov, -size, y, x)
                 for ov, size, y, x in targets),
            )
            clicks = [click_key(x, y) for _, _, _, y, x in ranked]
            if tier > 0:
                assert self.perception is not None
                board = self.perception.board_mask
                seen = set(clicks)
                lattice: list[ActionKey] = []
                for step in self.cfg.lattice_steps[:tier]:
                    for y in range(step // 2, board.shape[0], step):
                        for x in range(step // 2, board.shape[1], step):
                            ck = click_key(x, y)
                            if board[y, x] and ck not in seen:
                                seen.add(ck)
                                lattice.append(ck)
                clicks += sorted(lattice, key=lambda a: tries[a])
        arch = self.archetype
        if arch == "avatar":
            menu = dirs + aux + clicks
        elif arch == "click":
            menu = aux + clicks + dirs
        else:  # mixed / unknown
            menu = dirs + aux + clicks
        # A-not-B, globalized: an action measured null at nearly every state
        # (ft09: 3189/3561 tried edges were self-loops) is a last resort
        # everywhere, not a per-state rediscovery. Demotion, not deletion —
        # completeness is preserved, the sweep order just stops paying the
        # dead cells first.
        nulls, tries = self.action_nulls, self.action_tries
        dead_set = {a for a in menu
                    if tries[a] >= 6 and nulls[a] / tries[a] >= 0.9}
        # T2 click-null generalization: colors whose clicks a GATED rule
        # predicts null are demoted for ALL coordinates of that color —
        # unlike the per-key counter above, this covers never-tried cells
        # (ft09-class waste). Demotion, not deletion: completeness holds.
        if engine is not None and 6 in avail:
            null_colors = {
                r.color for r in engine.gated_rules()
                if getattr(r, "family", "") == "click_null"
            }
            if null_colors:
                assert self.perception is not None
                masked = self.perception.mask_frame(
                    np.asarray(node.frame, dtype=np.int16))
                for a in menu:
                    if a[0] == 6 and a not in dead_set:
                        if int(masked[a[2], a[1]]) in null_colors:
                            dead_set.add(a)
        keep = [a for a in menu if a not in dead_set]
        dead = [a for a in menu if a in dead_set]
        menu = keep + dead
        self._menu_cache[cache_key] = menu
        return menu

    # -- stepping ----------------------------------------------------------

    def _step(self, renv: RecordingEnv, akey: ActionKey) -> Any:
        from arcengine import GameAction

        aid = akey[0]
        if aid == 0:
            return renv.step(GameAction.RESET)
        if aid == 6:
            return renv.step(GameAction.ACTION6,
                             data={"x": int(akey[1]), "y": int(akey[2])})
        return renv.step(GameAction.from_id(aid))

    # -- battery interleave ------------------------------------------------

    def _fresh_battery_state(self, obs: Any, bcfg: BatteryConfig) -> BatteryState:
        """A BatteryState rooted at the CURRENT obs — no env.reset() (a reset
        mid-game costs a scored action and replays the level)."""
        gid = obs.game_id or "unknown"
        avail = sorted(a for a in (obs.available_actions or []) if a)
        st = BatteryState(game_id=gid, stem=game_stem(gid), avail=avail)
        dir_ids = [a for a in (1, 2, 3, 4) if a in avail]
        reps = bcfg.directional_reps
        if dir_ids and 6 not in avail:
            reps = max(reps, bcfg.budget // len(dir_ids))
        st.dir_queue = [a for a in dir_ids for _ in range(reps)]
        if not st.dir_queue:
            st.aux_queue = [a for a in bcfg.aux_action_ids if a in avail
                            for _ in range(bcfg.aux_reps_base)]
        st.prev_frame = settled_frame(obs).tolist()
        return st

    def _absorb_profile(self, profile: PerceptionProfile) -> None:
        """Update priors from a (re-)probe. The game-canonical Perception is
        set once, from the first battery (Stage-1a validated it per game);
        re-probes only refresh REACTIVE priors and an unknown archetype."""
        mask = profile.reactive_mask.astype(bool)
        for c in profile.clicks:
            if c["changed"]:
                mask[c["y"], c["x"]] = True
        if mask.any() or self.reactive_mask is None:
            self.reactive_mask = mask
        if self.archetype in ("", "unknown"):
            self.archetype = profile.archetype
        self._menu_cache.clear()

    # -- the play loop -----------------------------------------------------

    def play(self, env: Any, game_id: str = "") -> GameReport:
        from arcengine import GameState

        cfg = self.cfg
        t0 = time.monotonic()

        # per-game state
        self.perception: Perception | None = None
        self.graphs: dict[int, LevelGraph] = {}
        self.level_actions: dict[int, int] = defaultdict(int)
        self.reactive_mask: np.ndarray | None = None
        self.archetype = ""
        self.max_level = 0
        self.plan_mismatches = 0
        self.effects_engine: EffectEngine | None = None
        self.rule_mispredictions = 0
        self.predicted_steps = 0
        self.audit_plans = 0
        self.win_pred_attempts = 0
        self.win_pred_hits = 0
        self._plans_since_audit = 0
        self.action_tries: Counter = Counter()
        self.action_nulls: Counter = Counter()
        self._pending: list[_RawT] = []
        self._menu_cache: dict[Any, list[ActionKey]] = {}
        self._cur_key: NodeKey | None = None
        self.stem = game_stem(game_id) if game_id else "?"

        renv = RecordingEnv(env, self)
        battery = ProbeBattery(BatteryConfig(budget=cfg.battery_budget))
        profile, bstate = battery.run(renv)  # calls renv.reset() once
        self.stem = profile.stem
        self.perception = profile.perception()
        if cfg.effects_enabled:
            self.effects_engine = EffectEngine(self.perception,
                                               cfg.effect_config)
        self._absorb_profile(profile)
        self.archetype = profile.archetype
        battery_actions = bstate.actions_spent
        self._replay_pending()  # battery transitions also train the rules

        obs = renv.last_obs
        win_levels = getattr(obs, "win_levels", None)
        end_reason = "unknown"
        won = False
        tier = 0
        last_level = int(obs.levels_completed or 0)

        while True:
            if str(obs.state) == str(GameState.WIN):
                won, end_reason = True, "win"
                break
            if renv.actions >= cfg.budget:
                end_reason = "budget"
                break
            if time.monotonic() - t0 > cfg.wall_s:
                end_reason = "wall_clock"
                break
            if str(obs.state) == str(GameState.GAME_OVER):
                obs = self._step(renv, RESET_ACTION)
                continue

            level = int(obs.levels_completed or 0)
            if level != last_level:
                last_level = level
                tier = 0
                if self.effects_engine is not None:
                    self.effects_engine.on_level_up()
                if cfg.reprobe_on_level_up:
                    # the re-probe spends scored actions too: never past the
                    # per-game budget (measured +7 overshoot on tu93)
                    left = max(cfg.budget - renv.actions, 0)
                    bcfg = BatteryConfig(budget=min(cfg.reprobe_budget, left))
                    st = self._fresh_battery_state(obs, bcfg)
                    # pass the pre-rooted state: run(state=None) would call
                    # env.reset() and replay the level
                    rp, _ = ProbeBattery(bcfg).run(renv, state=st)
                    self._absorb_profile(rp)
                    obs = renv.last_obs
                    continue

            mult = cfg.level_stop_multiplier
            apl = cfg.human_apl.get(self.stem)
            if mult is not None and apl:
                if self.level_actions[level] > mult * apl:
                    end_reason = "stop_loss"
                    break

            g = self._graph(level)
            if self._cur_key is None or self._cur_key not in g.nodes:
                frame = settled_frame(obs)
                key = g.effective_key(self.perception.state_key(frame), None)
                g.ensure_node(key, frame, tuple(
                    sorted(a for a in (obs.available_actions or []) if a)))
                g.set_root(key)
                self._cur_key = key

            menu_fn = lambda k, _g=g, _t=tier: self._menu(_g, k, _t)  # noqa: E731
            fx = (_PlannerEffects(self.effects_engine, g)
                  if self.effects_engine is not None else None)
            # Stage 2b audit share: every audit_every-th frontier plan must
            # probe a rule-predicted pair (reorder-not-prune; the ft09 fix)
            audit_due = (fx is not None and fx.active
                         and self._plans_since_audit
                         >= cfg.effect_config.audit_every)
            p: Plan = plan(g, self._cur_key, menu_fn,
                           prior_penalty=cfg.prior_penalty,
                           greedy_rank_max=cfg.greedy_rank_max,
                           effects=fx, audit=audit_due)
            if p.kind == "none":
                if tier < len(cfg.lattice_steps):
                    tier += 1
                    continue
                end_reason = "frontier_exhausted"
                break
            if p.kind == "frontier":
                if p.audit:
                    self.audit_plans += 1
                    self._plans_since_audit = 0
                else:
                    self._plans_since_audit += 1

            src_key = self._cur_key  # source of step i (for rule demotion)
            executed_last = False
            for i, akey in enumerate(p.actions):
                step_predicted = bool(p.predicted) and i < len(p.predicted) \
                    and p.predicted[i]
                obs = self._step(renv, akey)
                if step_predicted:
                    self.predicted_steps += 1
                if i == len(p.actions) - 1:
                    executed_last = True
                if str(obs.state) != str(GameState.NOT_FINISHED):
                    break
                if int(obs.levels_completed or 0) != level:
                    break
                if renv.actions >= cfg.budget:
                    break
                if i < len(p.expected) and self._cur_key != p.expected[i]:
                    self.plan_mismatches += 1
                    if step_predicted and self.effects_engine is not None:
                        # a rule-proposed edge failed live: demote the rule,
                        # flag the source state back to the Markov machinery
                        self.effects_engine.live_mispredict(src_key, akey)
                        self.rule_mispredictions += 1
                    break
                src_key = p.expected[i] if i < len(p.expected) else self._cur_key

            # Stage 2b: a predicted-win attempt is verified by the env —
            # success = the final action leveled up (or won the game)
            if p.kind == "win_pred" and executed_last \
                    and self.effects_engine is not None:
                success = (int(obs.levels_completed or 0) > level
                           or str(obs.state) == str(GameState.WIN))
                self.effects_engine.win_attempt_result(p.actions[-1], success)
                self.win_pred_attempts += 1
                self.win_pred_hits += int(success)

        levels = self.max_level
        if won and win_levels:
            levels = int(win_levels)
        return GameReport(
            stem=self.stem,
            game_id=getattr(obs, "game_id", game_id) or game_id,
            won=won,
            levels_completed=levels,
            win_levels=win_levels,
            actions_total=renv.actions,
            per_level_actions=dict(self.level_actions),
            battery_actions=battery_actions,
            plan_mismatches=self.plan_mismatches,
            violation_events=sum(g.violation_events for g in self.graphs.values()),
            violated_states=sum(len(g.violated) for g in self.graphs.values()),
            end_reason=end_reason,
            wall_s=round(time.monotonic() - t0, 2),
            archetype=self.archetype,
            graph_stats={lvl: g.stats() for lvl, g in sorted(self.graphs.items())},
            effect_stats=(self.effects_engine.stats_rows()
                          if self.effects_engine is not None else []),
            rule_mispredictions=self.rule_mispredictions,
            predicted_steps_taken=self.predicted_steps,
            audit_plans=self.audit_plans,
            win_pred_attempts=self.win_pred_attempts,
            win_pred_hits=self.win_pred_hits,
        )


if __name__ == "__main__":  # standalone: play one game
    import sys

    from engineered.envs import open_arcade, resolve_game_ids

    stems = sys.argv[1:] or ["sb26"]
    arcade = open_arcade()
    gid_of = resolve_game_ids(arcade)
    for stem in stems:
        env = arcade.make(game_id=gid_of[stem], scorecard_id=f"eng1b-{stem}")
        agent = EngineeredAgent(AgentConfig(budget=4000, verbose=True))
        r = agent.play(env, gid_of[stem])
        print(f"{r.stem}: levels={r.levels_completed}/{r.win_levels} won={r.won} "
              f"actions={r.actions_total} apl={r.apl_mean} end={r.end_reason} "
              f"viol={r.violation_events} wall={r.wall_s}s")
