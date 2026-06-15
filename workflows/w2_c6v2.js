// W2-C6v2 — focused redesign of C6 (planner) to plan over the C4 forward model.
// Run: Workflow({ scriptPath: "<this file>" })

export const meta = {
  name: 'arcagi3-c6-planner-v2',
  description: 'Redesign+implement C6 planner over the C4 forward model (judge-gated, not-inert)',
  phases: [
    { title: 'Implement', detail: 'plan over C4 forward model (opus)' },
    { title: 'Review', detail: 'judge: not inert + no regression (opus)' },
    { title: 'Measure', detail: 'offline gate via uv' },
  ],
}

const REPO = '/Users/ahmed/Documents/ArcAGI3'

const VERDICT = {
  type: 'object',
  properties: {
    pass: { type: 'boolean' },
    not_inert: { type: 'boolean', description: 'does the planner emit >0 plan actions + reach goal on >=3 local games (the prior failure)?' },
    no_occ_gate: { type: 'boolean', description: 'planner activation does NOT depend on occ.usable?' },
    default_off_ok: { type: 'boolean' },
    feedback: { type: 'string' },
  },
  required: ['pass', 'not_inert', 'no_occ_gate', 'feedback'],
}
const MEASURE = {
  type: 'object',
  properties: {
    tests_pass: { type: 'boolean' },
    local_levels: { type: 'integer' },
    push_levels: { type: 'integer' },
    planner_emitted_actions: { type: 'boolean', description: 'did the planner (flag ON) emit >0 actions in a quick local check?' },
    summary: { type: 'string' },
  },
  required: ['tests_pass', 'local_levels', 'push_levels', 'summary'],
}

const impl = await agent(
  `Re-implement component C6 (planner) for the ARC-AGI-3 rebuild. It FAILED TWICE because it grounded on the occupancy map (occ.usable needs both movement axes, which is often unavailable -> planner 100% inert).\nREAD (mandatory): ${REPO}/rebuild/design/C6_v2.md (the REDESIGN — plan over the C4 forward model, NOT occupancy), ${REPO}/rebuild/design/C6_feedback.md (both prior failures), ${REPO}/src/arcagi3/forward_model.py (C4 — your transition model: scene_from_grid, predict, rollout, verify_step), ${REPO}/src/arcagi3/goals.py (C5 goal hypotheses), ${REPO}/src/arcagi3/wm_policy.py (integration point), and policy.py/perception.py.\nImplement a bounded forward-model search planner (BFS/A* over predicted Scenes toward the C5 goal/subgoal), behind a default-OFF ARCAGI3_PLANNER flag, integrated into WorldModelPolicy. Add tests INCLUDING one asserting the planner emits >0 plan actions and reaches the goal-test on >=3 local games (collect/switchdoor/push/maze).\nUse the venv: \`cd ${REPO} && uv run pytest -q\` and \`cd ${REPO} && PYTHONPATH=src uv run python -m arcagi3.runner --agent reactive --budget 4000 --quiet\` (must stay 8/8, 27 levels, push 3/3). Do NOT gate planning on occ.usable. Return what you built + proof the planner is NOT inert (action counts).`,
  { label: 'impl:C6v2', phase: 'Implement', model: 'opus', agentType: 'general-purpose' }
)

const review = impl ? await agent(
  `Adversarially review the C6 v2 planner. Read ${REPO}/rebuild/design/C6_v2.md + C6_feedback.md, inspect new code (\`cd ${REPO} && git add -N . && git diff\`). The prior two attempts were INERT (0 plan actions). VERIFY not_inert (run planner flag ON on local games and confirm >0 plan actions + goal reached on >=3 games), no_occ_gate (activation independent of occ.usable), default_off_ok (byte-identical OFF), and no push/local regression. Be ruthless about inertness.`,
  { label: 'review:C6v2', phase: 'Review', model: 'opus', schema: VERDICT, agentType: 'pr-review-toolkit:code-reviewer' }
) : null

const measure = (review && review.pass) ? await agent(
  `Measure C6 v2 NO-REGRESSION via venv. The numbers MUST come from the DEFAULT REACTIVE agent (NO env flags):\n  cd ${REPO} && uv run pytest -q\n  cd ${REPO} && PYTHONPATH=src uv run python -m arcagi3.runner --agent reactive --budget 4000 --quiet\nReport tests_pass, and from THAT reactive run: local TOTAL levels (expect 27, which already includes push 3/3) and the push game's levels (expect 3). Do NOT report the wm/planner run's numbers for these fields.`,
  { label: 'measure:C6v2', phase: 'Measure', model: 'haiku', schema: MEASURE, agentType: 'general-purpose' }
) : null

// reactive local_levels==27 already requires push 3/3 (else total<27), so don't separately
// gate on the (previously mis-sourced) push field.
const ok = review && review.pass && review.not_inert && review.no_occ_gate &&
           measure && measure.tests_pass && measure.local_levels >= 27
if (ok) {
  await agent(`Commit: \`cd ${REPO} && git add -A && git commit -q -m "feat(rebuild): C6 planner v2 (plans over C4 forward model; not inert; judge-gated)"\`; show \`git log --oneline -1\`.`,
    { label: 'commit:C6v2', phase: 'Measure', model: 'haiku', agentType: 'general-purpose' })
  log('C6 v2 PASSED + committed (planner active).')
} else {
  await agent(`Discard failed C6 v2: \`cd ${REPO} && git checkout -- src tests 2>/dev/null; git clean -fdq src tests scripts\`; confirm \`uv run pytest -q\` green + clean status.`,
    { label: 'revert:C6v2', phase: 'Measure', model: 'haiku', agentType: 'general-purpose' })
  log(`C6 v2 FAILED -> reverted. ${review ? 'not_inert=' + review.not_inert + ' ' + review.feedback : 'impl/review failed'}`.slice(0, 300))
}

return { status: ok ? 'passed' : 'failed', review, measure }
