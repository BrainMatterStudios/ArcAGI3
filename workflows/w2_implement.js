// W2 — ARC-AGI-3 rebuild IMPLEMENTATION workflow (judge gate + model tiering).
//
// Run AFTER: (a) submission score reviewed, (b) W1 design blueprints exist at
// rebuild/design/<Cx>.md. Invoke with:
//   Workflow({ scriptPath: "<this file>" })
//
// Design: components are implemented SEQUENTIALLY in dependency order (each builds on the
// previous merged code), each gated by an adversarial judge AND an offline measurement.
// Edits land in the MAIN repo (sequential deps preclude worktree isolation); a component
// that fails the gate is reverted (git checkout) and its judge feedback recorded — it does
// NOT block later independent components. Real-game (NORMAL-mode) A/B is run by the human
// operator after W2, since it needs the gitignored ARC_API_KEY and is slow; W2's gate uses
// the fast offline suite (pytest + local 8/8 + push 3/3) for no-regression.
//
// Model tiering: high-complexity components -> opus; medium -> sonnet; judge -> opus;
// measurement/triage -> haiku.

export const meta = {
  name: 'arcagi3-rebuild-implement',
  description: 'Implement the ARC-AGI-3 world-model rebuild components, judge-gated + measured',
  phases: [
    { title: 'Implement', detail: 'write each component per its blueprint (model by tier)' },
    { title: 'Review', detail: 'adversarial judge reviews the diff vs blueprint + no-regression (opus)' },
    { title: 'Measure', detail: 'offline gate: pytest + local 8/8 + push 3/3 (haiku)' },
  ],
}

const REPO = '/Users/ahmed/Documents/ArcAGI3'

// build order respects dependencies C1->C2->C3->(C4,C5)->C6->C7
const ORDER = [
  { id: 'C1', tier: 'medium' },
  { id: 'C2', tier: 'high' },
  { id: 'C3', tier: 'high' },
  { id: 'C4', tier: 'high' },
  { id: 'C5', tier: 'high' },
  { id: 'C6', tier: 'high' },
  { id: 'C7', tier: 'medium' },
]
const MODEL = { high: 'opus', medium: 'sonnet' }

const VERDICT = {
  type: 'object',
  properties: {
    pass: { type: 'boolean' },
    regression_risk: { type: 'string' },
    issues: { type: 'array', items: { type: 'string' } },
    feedback: { type: 'string' },
  },
  required: ['pass', 'feedback'],
}
const MEASURE = {
  type: 'object',
  properties: {
    tests_pass: { type: 'boolean' },
    local_levels: { type: 'integer', description: 'total local levels (expect 27)' },
    push_levels: { type: 'integer', description: 'push levels (expect 3)' },
    summary: { type: 'string' },
  },
  required: ['tests_pass', 'local_levels', 'push_levels', 'summary'],
}

const GUARD = `NO-REGRESSION IS MANDATORY. After your change, \`uv run pytest\` must stay green and
\`PYTHONPATH=src uv run python -m arcagi3.runner --agent reactive --budget 4000 --quiet\` must still
show 8/8 wins, 27 total levels, push 3/3. New behaviour MUST be behind a flag/selector defaulting
OFF (the submission agent must be unchanged) unless it strictly dominates. If you cannot meet this,
STOP and report — do not ship a regression.`

const results = []
for (const c of ORDER) {
  phase('Implement')
  const impl = await agent(
    `Implement component ${c.id} of the ARC-AGI-3 rebuild.\nREAD: ${REPO}/rebuild/design/${c.id}.md (the judged blueprint), ${REPO}/REBUILD_PLAN.md, and the relevant src/arcagi3/*.py.\nWrite the code + offline unit tests following the blueprint's implementation_checklist. Match the existing code style.\n${GUARD}\nReturn a concise summary of exactly what files/functions you added/changed and how it's gated.`,
    { label: `impl:${c.id}`, phase: 'Implement', model: MODEL[c.tier], agentType: 'general-purpose' }
  )
  if (!impl) { results.push({ id: c.id, status: 'impl-failed' }); continue }

  phase('Review')
  const review = await agent(
    `Adversarially review the implementation of ${c.id}. Run \`cd ${REPO} && git diff\` to see the change. Check: (1) matches ${REPO}/rebuild/design/${c.id}.md; (2) correctness/edge cases; (3) NO-REGRESSION — is new behaviour gated OFF by default / does it risk the sokoban (push) or local suite? (4) generality to unseen games. Be ruthless; this project has repeatedly regressed push.`,
    { label: `review:${c.id}`, phase: 'Review', model: 'opus', schema: VERDICT, agentType: 'pr-review-toolkit:code-reviewer' }
  )

  phase('Measure')
  const measure = await agent(
    `Measure no-regression for ${c.id}. Run:\n  cd ${REPO} && uv run pytest -q\n  cd ${REPO} && PYTHONPATH=src uv run python -m arcagi3.runner --agent reactive --budget 4000 --quiet\nReport whether tests pass and the local TOTAL levels (expect 27) and push levels (expect 3).`,
    { label: `measure:${c.id}`, phase: 'Measure', model: 'haiku', schema: MEASURE, agentType: 'general-purpose' }
  )

  const ok = review && review.pass && measure && measure.tests_pass &&
             measure.local_levels >= 27 && measure.push_levels >= 3
  if (!ok) {
    // revert ONLY this component's (uncommitted) changes; prior passed components are
    // already committed, so checkout+clean of src/tests cannot touch them.
    await agent(
      `Discard the uncommitted changes from the failed ${c.id} implementation so they cannot poison later components: run exactly \`cd ${REPO} && git checkout -- src tests 2>/dev/null; git clean -fdq src tests\`. Then confirm \`cd ${REPO} && uv run pytest -q\` is green and \`git status --short\` is clean. Report the final git status.`,
      { label: `revert:${c.id}`, phase: 'Measure', model: 'haiku', agentType: 'general-purpose' }
    )
    results.push({ id: c.id, status: 'gate-failed', review, measure })
    log(`${c.id} FAILED gate -> reverted. feedback: ${review ? review.feedback : 'n/a'}`)
    continue
  }
  // commit the passed component so later components build on it and reverts stay scoped
  await agent(
    `Commit the passed ${c.id} implementation: run \`cd ${REPO} && git add -A && git commit -q -m "feat(rebuild): ${c.id} (W2 judge-gated, no-regression verified)"\`. Confirm with \`git log --oneline -1\`.`,
    { label: `commit:${c.id}`, phase: 'Measure', model: 'haiku', agentType: 'general-purpose' }
  )
  results.push({ id: c.id, status: 'passed', impl, review, measure })
  log(`${c.id} PASSED gate + committed (local ${measure.local_levels} levels, push ${measure.push_levels}).`)
}

return { results, passed: results.filter((r) => r.status === 'passed').map((r) => r.id) }
