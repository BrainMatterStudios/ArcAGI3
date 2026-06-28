"""ChainMacroExplorer — within-game SOLUTION-SEQUENCE replay (geodesic-replay / CausalChainMacro).

The RHAE headroom oracle (scripts/rhae_headroom.py) measured 35x MEDIAN (100-800x on L2+) recoverable
within-game efficiency: the explorer re-discovers the same mechanic from scratch every level. TransferExplorer
captures only a sliver of it -- it promotes the rewarding action's CLASS to tier 0 but the explorer still
EXPLORES to find where to apply it, in arbitrary order. The headroom is the multi-step SOLUTION SEQUENCE.

This subclass records, per level, the ordered sequence of EFFECTIVE clicks (a click that changed the state
key -> a real mechanic operation, not a no-op) keyed by the same transferable signature TransferExplorer
learns (object colour). On the next level it REPLAYS that chain as a directed EXPLOIT: for each chain step it
clicks an as-yet-unclicked object whose signature matches, in order. For CLICK/recolor games (where the
headroom is largest: lf52/vc33/cd82/ar25) the matching object is clickable from ANY state, so the chain
replays with NO navigation -- sidestepping the spatial-nav wall that killed every prior planner. If the chain
can't progress (no matching object, or it runs out without a level-up) it hands straight back to the base
explorer, so coverage is preserved and a misfire only costs the (short) chain length.

This is EXPLOIT-toward-the-known-reward-chain (driven by the previous levels' real level-up edges), not a
signal-free frontier reorder -- the W3-safe class. enable_macro=False -> byte-identical to TransferExplorer
(the banked submission), the firewall. See docs .../2026-06-21-history-augmented-state-design.md and the
memory arcagi3-rhae-headroom.
"""
from __future__ import annotations

from . import perception as P
from .events import EventExtractor
from .transfer_explorer import TransferExplorer


class ChainMacroExplorer(TransferExplorer):
    def __init__(self, *args, enable_macro: bool = True, macro_mode: str = "gated",
                 gate_thresh: float = 0.9, gate_probe_k: int = 4, min_chain_sigs: int = 2,
                 **kwargs) -> None:
        self.enable_macro = bool(enable_macro)
        # "gated" (default, STRICT-SUPERSET attempt): on a new level, DON'T replay yet -- explore normally
        #   while gathering this level's early effective-click signatures, then deploy the HARD chain replay
        #   ONLY if those signatures match the chain's (jaccard >= gate_thresh) i.e. the mechanic is STABLE.
        #   Mechanic-shift games (cd82/vc33: jaccard 0.25-0.33) never deploy -> no derail; stable games
        #   (lp85: jaccard 1.00) deploy -> capture the headroom. The probe is just normal exploration (never
        #   wasted), so it is strict-superset by construction. gate_probe_k = effective clicks gathered
        #   before deciding; gate_thresh = jaccard cutoff.
        # "hard": directed object-click override -> fast (lp85 21x) but DERAILS shifting levels (cd82/vc33).
        # "soft": coverage-safe ordered tier-0 promotion -> no derail but loses the gain (~= transfer).
        self.macro_mode = macro_mode
        self.gate_thresh = float(gate_thresh)
        self.gate_probe_k = int(gate_probe_k)
        # min distinct signatures the chain must have to be deployable. A single-signature chain (e.g.
        # vc33's all-colour-9 flood-fill buttons) is spatially-specific and does NOT transfer even when
        # the colour set matches across levels; requiring >=2 distinct colours restricts replay to genuine
        # multi-type "click these object-kinds" mechanics (lp85 {8,14}) that DO transfer.
        self.min_chain_sigs = int(min_chain_sigs)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.macro: list = []            # the learned ordered chain (click signatures)
        self.macro_pos: int = 0
        self.in_macro: bool = False
        self._level_ops: list = []       # effective-click signatures THIS level, in order
        self._macro_clicked: set = set()  # cells already clicked during the current macro replay
        self._cur_grid = None
        # gated mode: pre-flight probe state
        self._gate_pending: bool = False  # gathering this level's early sigs before deciding to deploy
        self._macro_sigs: set = set()     # COMPLETE effective-sig set of the PREVIOUS level (gate ref)
        self._level_sigs: set = set()     # current level's early effective sigs (probe, for the jaccard)
        self._level_sigset: set = set()   # current level's COMPLETE effective-sig set (banked at level-up)
        self._probe_eff: int = 0          # effective clicks gathered this level (probe progress)
        # typed_tripwire mode: per-step typed-effect verification
        self._ext = EventExtractor()
        self._level_typed: list = []      # typed effect-sig of each effective click THIS level (parallel _level_ops)
        self.macro_typed: list = []       # banked typed-effect sigs of the chain
        self._verify_pos: int = 0         # which banked typed-sig to verify next during replay

    def _typed_sig(self, action):
        """Position-free typed effect of a click: frozenset of (EventType, colour) over its salient events.
        Coarse enough to be stable when the same mechanic recurs (lp85), fine enough to flip when the
        mechanic shifts (vc33's colour-9 click does something different on a later level)."""
        if self._prev_grid is None or self._cur_grid is None:
            return frozenset()
        se = self._ext.extract(self._prev_grid, self._cur_grid, action, 0.0, bg=self.bg)
        return frozenset((e.type.name, int(e.color) if e.color is not None else -1) for e in se.salient)

    # --- learn the chain: effective clicks (state-changing) in order -----------------------------
    def _record(self, key, action, next_key, reward, cands, terminal):
        super()._record(key, action, next_key, reward, cands, terminal)
        if not (self.enable_macro and action[0] == "C" and self._prev_grid is not None):
            return
        # typed_tripwire: verify EVERY replayed click (incl. no-ops) -- abort on the first realized typed
        # effect that diverges from what the banked chain step produced (the mechanic shifted on this level).
        if self.macro_mode == "typed_tripwire" and self.in_macro:
            if (self._verify_pos >= len(self.macro_typed)
                    or self._typed_sig(action) != self.macro_typed[self._verify_pos]):
                self.in_macro = False
            self._verify_pos += 1
        if next_key == key or reward != 0:
            return
        sig = self._cell_to_sig(self._prev_grid).get((action[2], action[1]))
        if sig is None:
            return
        self._level_sigset.add(sig)                  # COMPLETE effective-sig set (gate reference, clean)
        if not self._level_ops or self._level_ops[-1] != sig:
            self._level_ops.append(sig)              # the actual working sequence this level
            if self.macro_mode == "typed_tripwire":
                self._level_typed.append(self._typed_sig(action))   # parallel typed effect
        # soft replay: advance the chain pointer when its current step is achieved
        if (self.macro_mode == "soft" and self.in_macro
                and self.macro_pos < len(self.macro) and sig == self.macro[self.macro_pos]):
            self._macro_clicked.add((action[2], action[1]))
            self.macro_pos += 1
        # gated pre-flight: gather this level's early effective sigs, then deploy iff stable
        if self.macro_mode == "gated" and self._gate_pending:
            self._level_sigs.add(sig)
            self._probe_eff += 1
            if self._probe_eff >= self.gate_probe_k:
                self._evaluate_gate()

    # --- on level-up: bank the chain and arm replay for the next level ---------------------------
    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_macro:
            if gstate_terminal or gstate_notplayed:
                self.in_macro = False
                self._level_ops = []
                self._level_sigset = set()
            elif levels > self.prev_levels:
                if self._level_ops:
                    self.macro = list(self._level_ops)   # this level's solution sequence
                    self.macro_typed = list(self._level_typed)
                self._level_ops = []
                self._level_typed = []
                self.macro_pos = 0
                self._verify_pos = 0
                self._macro_clicked = set()
                if self.macro_mode == "gated":
                    # arm the pre-flight probe: explore normally, gather sigs, deploy only if stable.
                    # gate reference = the COMPLETE effective-sig set of the level just solved (matches the
                    # clean offline probe; the deduped ordered `macro` undercounts under transfer-bias).
                    self._macro_sigs = set(self._level_sigset)
                    self._level_sigs = set()
                    self._probe_eff = 0
                    self._gate_pending = bool(self.macro)
                    self.in_macro = False
                else:
                    self.in_macro = bool(self.macro)
                self._level_sigset = set()
            self._cur_grid = grid
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _evaluate_gate(self):
        """Deploy the hard chain replay iff this level's early effective signatures match the chain's
        (mechanic stable). Otherwise stay banked (no replay) -> no derail. Strict-superset by construction:
        the probe was just normal exploration."""
        self._gate_pending = False
        if len(self._macro_sigs) < self.min_chain_sigs:
            return                          # single-signature chain = spatially-specific, won't transfer
        union = self._level_sigs | self._macro_sigs
        jac = (len(self._level_sigs & self._macro_sigs) / len(union)) if union else 0.0
        if jac >= self.gate_thresh:
            self.in_macro = True            # stable multi-type mechanic -> deploy the hard chain replay
            self.macro_pos = 0
            self._macro_clicked = set()

    def _candidates(self, grid, available):
        cands = super()._candidates(grid, available)   # TransferExplorer reward-class promotion first
        if (self.macro_mode != "soft" or not self.enable_macro or not self.in_macro
                or self.macro_pos >= len(self.macro)):
            return cands
        cell2sig = self._cell_to_sig(grid)
        present = set(cell2sig.values())
        while self.macro_pos < len(self.macro) and self.macro[self.macro_pos] not in present:
            self.macro_pos += 1                          # skip chain steps with no object on this grid
        if self.macro_pos >= len(self.macro):
            self.in_macro = False
            return cands
        target = self.macro[self.macro_pos]
        out = []                                         # ADDITIVE: promote the current step, demote nothing
        for (act, tier) in cands:
            if (act[0] == "C" and cell2sig.get((act[2], act[1])) == target
                    and (act[2], act[1]) not in self._macro_clicked):
                out.append((act, 0))
            else:
                out.append((act, tier))
        return out

    def _choose(self, cur):
        # gated mode deploys the SAME hard directed replay as "hard", but only after the pre-flight gate
        # opened (mechanic confirmed stable on this level), so it never derails a shifting level.
        if (self.macro_mode in ("hard", "gated", "typed_tripwire") and self.enable_macro and self.in_macro
                and self.macro_pos < len(self.macro)):
            a = self._macro_pick()
            if a is not None:
                return a
            self.in_macro = False   # chain can't progress here -> hand back to the base explorer
        return super()._choose(cur)

    def _macro_pick(self):
        """Click an as-yet-unclicked object matching the current chain step's signature (directed
        exploit). Returns a ("C", x=col, y=row) token, or None if no match (caller falls back)."""
        if self._cur_grid is None or self.macro_pos >= len(self.macro):
            return None
        target = self.macro[self.macro_pos]
        best = None
        for o in P.connected_components(self._cur_grid, background=self.bg):
            if self._click_sig(o) != target:
                continue
            cell = (int(round(o.centroid[0])), int(round(o.centroid[1])))   # (row, col)
            if cell not in self._macro_clicked:
                best = cell
                break
        if best is None:
            return None
        self._macro_clicked.add(best)
        self.macro_pos += 1
        return ("C", int(best[1]), int(best[0]))   # token is (col, row)
