# Design — ARC-AGI-3 Hybrid Explorer Agent

A general, **training-free** agent for ARC-AGI-3. Dropped into an unseen 64×64 grid game
with no instructions, no stated goal, and no rules, it must infer the goal from the
levels-completed reward, build a world model online, and clear levels — and crucially
**generalize to unseen private games** (no per-game code).

## Why training-free / search-based

The documented SOTA (graph-based exploration) is training-free and beat both an LLM+DSL
baseline and the CNN sample. The eval is offline (no hosted LLMs) on a 12h budget. A
self-contained, goal-directed search agent is the highest-leverage approach. The official
Kaggle sample (a CNN that rewards *any* frame change) scores only 0.25 — it explores, but
isn't goal-directed. Ours optimizes the actual reward (Δlevels_completed).

## Pipeline

```
frame (N,64,64) ─▶ Perception ─▶ object-centric state + masked state-hash
                                       │
                         World Model (directed state-transition graph)
                                       │
              Motion model (avatar + per-action displacement, action-correlated)
                                       │
        Reactive hybrid policy:  probe ─▶ navigate-to-goal-objects ─▶ graph fallback
                                       │
                                  GameAction
```

### Perception (`perception.py`)
- Connected-component objects (4-connectivity), background = most frequent color.
- **Volatility masking**: cells that change almost every step (counters) are masked from
  the state key so the graph doesn't explode.
- **Salient click targets**: object-centric click proposals (centroids/corners) by
  salience tier, instead of brute-forcing 4096 pixels; optional coarse-grid fallback for
  large click action-spaces.
- Exact masked state hashing for graph dedup.

### World model (`world_model.py`)
Directed graph: nodes = masked state keys, edges = (action → next state, reward). Drives
exploration (BFS to the nearest unexplored frontier) and exploitation (replay action
sequences that produced reward).

### Motion model (`movement.py`)
Detects the **controllable avatar** and learns each action's displacement by probing.
Key idea for robustness: the avatar's motion **correlates with the action** (distinct
deltas per action), whereas counters/animations move constantly — so the avatar is the
color with the most distinct delta vectors, and other movers are masked as distractors.
This converts exploration from blind state-graph BFS into cheap coordinate navigation.

### Reactive policy (`policy.py`)
One action per call (the official `choose_action` interface), state on the object:
1. **Probe** — try each simple action once; learn the motion model; identify animated
   distractors.
2. **Navigate** — drive the avatar to candidate goal objects in coordinate space; level-up
   is the reward signal.
3. **Graph fallback** — when no avatar or navigation stalls, explore the state graph
   (frontier search + shortest-path replay) and exploit known reward transitions.
RESET handles game-over; the whole thing degrades gracefully to safe exploration.

## Why it generalizes (evidence)

The *same* policy, with no per-game code, wins **8/8 local archetypes, 27/27 levels**
(4000-action budget): open navigation, walled maze, click-a-button, native-64 click
discrimination, animated-counter distractor, sokoban push, multi-subgoal collect, and
causal switch→door. Breadth across navigation / clicking / animation / causal / multi-goal
mechanics is the core requirement for the unseen private games.

## Known limitations / roadmap
- **Sokoban efficiency**: push solves but inefficiently (joint state-space exploration);
  a push-planner (detect pushable + plan pushes toward a marker) would cut actions.
- **Large state spaces**: graph-replay cost grows; replacing replay with coordinate-nav to
  frontier cells would scale avatar games further.
- **Validation gap**: tested only on self-authored games; real public-game validation
  (`validate_online.py`, needs an API key) is the next priority to avoid overfitting.

## Submission
`MyAgent(Agent)` adapter (`submission/my_agent.py`) wraps the policy in the official
ARC-AGI-3-Agents interface; `notebook.ipynb` runs it against the gateway-served games per
the official contract. See `submission/README.md`. MIT-0 licensed.
