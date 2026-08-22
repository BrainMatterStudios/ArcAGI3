# archetype_triage — envelope arithmetic (law #3)

This graft strictly RECLAIMS wall time; it cannot extend anything.

## Only-earlier-stop, by construction

- The single patched seam is `_HarnessGameSession.should_stop`
  (solver.py:246-261). The wrapper runs the STOCK verdict first, unguarded:
  every stock True stays True. Graft logic can only ADD a True — i.e. stop a
  session earlier — never flip a stock True to False or delay one.
- No sleeps, no retries, no timeout changes, no extra decode obligations:
  `max_runtime_s_per_game`, the analyzer timeout, the retry cadence, and
  the yield machinery are untouched. The graft's own work per call is a set
  lookup + two float compares.
- A fail-open error path returns the stock verdict (False) — behavior
  degrades to exactly stock, which is the 132-min clock.

## What a kill reclaims

- Killing a zero-level session at AVATAR 60m / MIXED 70m returns its worker
  thread and semaphore slot (solver.py:884-887, :896-917) to the queue,
  so a queued game starts up to 72/62 minutes earlier than the stock
  132-min exhaustion.
- Measured expectation (trigger_rule.json, pooled corpus n=103 sessions,
  110-game geometry): ~41 kills, ~2,932 reclaimed worker-minutes
  (~48.9 h, ~20% of total budget) per run, with 0 observed false negatives.
- The kill banks the session through the stock give-up path
  (`finish_game` -> state "gave_up", taaf/game.py:596-632): completed
  levels would keep their score — and the rule only ever fires on sessions
  with ZERO completed levels, so nothing scoreable is ever discarded.

## What it can cost (bounded by the trigger table)

- The only risk is a false negative: a zero-level session killed before a
  level it would eventually have completed. The thresholds are set from the
  latest observed first-completions (AVATAR 54.5m -> kill 60; MIXED 56.9m
  -> kill 70; CLICK tail reaches 122.6m -> never kill). AVATAR's 0/13 at
  60 carries a rule-of-three 95% upper bound of ~23% — the reason CLICK,
  where late completions are common and high-value (lp85: 88m, score 2.01),
  is never killed.

## Flags

- `TRIAGE_KILL=0` — disables entirely (install-time skip AND call-time
  pass-through).
- `TRIAGE_AVATAR_MIN` / `TRIAGE_MIXED_MIN` / `TRIAGE_CLICK_MIN` — per-
  archetype threshold in minutes; empty keeps the default (60 / 70 /
  never); `never`/`none`/`off`/`<=0` disables that archetype's kill.
- No TimeBank / grant machinery by design: pure kill-rule, single-variable.
