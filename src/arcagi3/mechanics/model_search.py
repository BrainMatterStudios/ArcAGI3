"""Model-beam search over executable mechanics (winning/mechanic-model-search, Phase 3/4).

Maintains a BEAM of executable WorldModels fit from the probe so far (never commits to one early),
MDL-scored, plus active information-gain probing (pick the action whose predicted outcome most
SEPARATES the beam, not the action that explores the most map). Goal hypotheses are enumerated
fresh from the scene and treated as candidates for the planner.

MDL objective (mission Phase 3):
    score = 5.0*transition_accuracy + 2.0*reward_event_accuracy
          - 0.5*num_primitives - 0.2*num_params - 1.0*contradiction_count
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from arcagi3.mechanics.primitives import enumerate_goals
from arcagi3.mechanics.world_model import WorldModel, _centroid, fit_world_model


def _mdl(model: WorldModel, test, reward_acc: float, contradictions: int) -> float:
    return (5.0 * model.transition_accuracy + 2.0 * reward_acc
            - 0.5 * model.num_primitives() - 0.2 * model.num_params() - 1.0 * contradictions)


def _held_out_accuracy(model, test):
    if not test:
        return 0.0
    return float(np.mean([model.score_transition(p, a, g) for p, a, g in test]))


def _held_out_move_acc(model, test):
    scores = [model.score_move(p, a, g) for p, a, g in test]
    scores = [s for s in scores if s is not None]
    return float(np.mean(scores)) if scores else 0.0


class ModelSearch:
    """Accumulates transitions, fits + ranks a model beam, proposes info-gain probes and goals."""

    def __init__(self, beam_k=16, width=64, height=64):
        self.beam_k = beam_k
        self.width, self.height = width, height
        self.transitions: list = []         # (prev_grid, action_id|None, grid)
        self.beam: list[WorldModel] = []
        self.reward_acc = 0.0

    def observe(self, prev_grid, action_id, grid):
        if prev_grid is not None:
            self.transitions.append((prev_grid, action_id, grid))

    # ---- fitting --------------------------------------------------------------
    def fit(self):
        """Refit the beam from all transitions. Builds a few executable candidates (primary,
        no-push, second-agent-candidate), scores each by MDL on a held-out split, keeps top-K."""
        n = len(self.transitions)
        if n < 6:
            self.beam = []
            return self.beam
        split = max(4, int(n * 0.7))
        train, test = self.transitions[:split], self.transitions[split:]
        candidates = []
        primary = fit_world_model(train, self.width, self.height)
        if primary is not None:
            candidates.append(primary)
            if primary.push.block_colors:                      # variant without push
                import copy
                nop = copy.deepcopy(primary); nop.push.block_colors = set()
                candidates.append(nop)
        scored = []
        for m in candidates:
            m.transition_accuracy = _held_out_accuracy(m, test or train)
            m.move_accuracy = _held_out_move_acc(m, test or train)
            contradictions = sum(1 for p, a, g in train if a in m.move.deltas
                                 and m.score_transition(p, a, g) < 0.34)
            scored.append((_mdl(m, test, self.reward_acc, contradictions), m))
        scored.sort(key=lambda x: -x[0])
        self.beam = [m for _s, m in scored[:self.beam_k]]
        return self.beam

    def best(self):
        return self.beam[0] if self.beam else None

    def confidence(self):
        # planning relies on the MOVEMENT model (paint prediction can be noisy yet plans still hold)
        return self.beam[0].move_accuracy if self.beam else 0.0

    # ---- active probing (Phase 4) --------------------------------------------
    def probe_action(self, grid, available):
        """Pick the action that maximally SEPARATES the beam's predicted agent moves (info gain).
        Falls back to the least-tried simple action when the beam can't discriminate."""
        simple = [a for a in available if a in (1, 2, 3, 4)]
        if not simple:
            return None
        if len(self.beam) < 2:
            # bootstrap: round-robin the simple actions to build the movement model
            tried = Counter(a for _p, a, _g in self.transitions if a is not None)
            return min(simple, key=lambda a: tried.get(a, 0))
        best_a, best_disagree = simple[0], -1
        for a in simple:
            preds = set()
            for m in self.beam:
                ap = m.agent_pos(grid)
                preds.add(m.predict_move(grid, ap, a))
            if len(preds) > best_disagree:
                best_disagree = len(preds); best_a = a
        return best_a

    # ---- goals ----------------------------------------------------------------
    def goals(self, grid):
        m = self.best()
        if m is None:
            return []
        objs = [int(c) for c in np.unique(grid)]
        return enumerate_goals(grid, m.agent_colors, m.bg, objs)
