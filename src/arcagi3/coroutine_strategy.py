"""CoroutineStrategy — drive a GENERATOR-style solver over the reactive decide() interface.

A generator solver is written in synchronous style: `obs = yield token` emits a token and receives the resulting
observation dict {grid, terminal, notplayed, levels, available}. This adapter bridges it to the portfolio's
one-token-per-call decide() contract, so the search-replay paradigm (reset/step to search a dirty run, then
double-reset + replay a clean run) can be written as ordinary control flow instead of a hand-rolled state machine.

Also provides `general_agent_gen`: a game-agnostic solver (single-click + two-phase select->apply affordance
probing, then affordance-pruned BFS, then clean replay) as such a generator. Abstains (benign) when exhausted.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from arcagi3 import perception as P
from arcagi3.general_search_strategy import _salient


class CoroutineStrategy:
    def __init__(self, gen_factory, seed: int = 0):
        self.gs = None
        self._factory = gen_factory
        self._gen = None
        self._done = False

    def _benign(self, available):
        for a in (5, 1, 2, 3, 4):
            if a in available:
                return ("S", a)
        return ("reset",)

    def decide(self, grid, gstate_terminal=False, gstate_notplayed=False, levels=0, available=()):
        available = list(available or [])
        obs = {"grid": grid, "terminal": bool(gstate_terminal), "notplayed": bool(gstate_notplayed),
               "levels": int(levels or 0), "available": available}
        if self._done:
            return self._benign(available)
        try:
            if self._gen is None:
                self._gen = self._factory(obs)
                tok = next(self._gen)           # prime to first yield (computed from obs0)
            else:
                tok = self._gen.send(obs)
            return tok if tok is not None else self._benign(available)
        except StopIteration:
            self._done = True
            return self._benign(available)


def _delta(a, b):
    return int(np.sum(a[:56, :56] != b[:56, :56]))


def general_agent_gen(obs0):
    """game-agnostic search-replay solver as a generator (frames+feedback only, zero per-game code)."""
    g0 = obs0["grid"]; avail = list(obs0["available"])
    macros = [[("S", a)] for a in avail if a in (1, 2, 3, 4, 5)]
    targets = _salient(g0, 16) if 6 in avail else []

    def replay_clean(seq):
        # double-reset -> fresh scored run, then execute the winning sequence
        yield ("reset",); yield ("reset",)
        for m in seq:
            for tok in m:
                yield tok

    # ---- affordance probing (single-click + two-phase select->apply) ----
    base = {}
    if 6 in avail:
        for (cx, cy) in targets:
            yield ("reset",)
            obs = yield ("C", cx, cy)
            if obs["levels"] >= 1:
                yield from replay_clean([[("C", cx, cy)]]); return
            base[(cx, cy)] = 0 if obs["terminal"] else _delta(obs["grid"], g0)
            if not obs["terminal"] and base[(cx, cy)] >= 2:
                macros.append([("C", cx, cy)])
        sels = [t for t in targets if base.get(t, 99) <= 3][:7]
        for s in sels:
            for t in targets[:10]:
                if t == s:
                    continue
                yield ("reset",)
                obs = yield ("C", *s)
                if obs["terminal"] or obs["levels"] >= 1:
                    continue
                b = obs["grid"]
                obs = yield ("C", *t)
                d = 0 if obs["terminal"] else _delta(obs["grid"], b)
                if obs["levels"] >= 1 or abs(d - base.get(t, 0)) >= 2:
                    macros.append([("C", *s), ("C", *t)])

    # ---- affordance-pruned BFS with clean replay on win ----
    seen = set(); q = deque([[]]); nodes = 0
    while q and nodes < 6000:
        seq = q.popleft()
        obs = yield ("reset",)          # capture the root (post-reset) state so the empty seq expands
        won = False; dead = False
        for m in seq:
            for tok in m:
                obs = yield tok
                nodes += 1
                if obs["levels"] >= 1:
                    won = True; break
                if obs["terminal"]:
                    dead = True; break
            if won or dead:
                break
        if won:
            yield from replay_clean(seq); return
        if obs is not None and not dead:
            st = hash(obs["grid"][:56, :56].tobytes())
            if st not in seen:
                seen.add(st)
                for m in macros:
                    q.append(seq + [m])
    while True:                     # exhausted -> abstain (benign, floor-safe)
        yield ("S", avail[0] if avail else 5)
