// W2-RETRY — re-implement the W2 components that failed their judge gate (C4, C6),
// feeding each implementer BOTH its blueprint and the prior judge feedback (the specific
// bugs to fix). Same gate: implement(opus) -> adversarial judge(opus) -> offline measure
// (uv) -> commit-on-pass / revert+clean-on-fail. Sequential: C4 (forward model) before
// C6 (planner), since C6 consumes C4.
//
// Run: Workflow({ scriptPath: "<this file>" })

export const meta = {
  name: 'arcagi3-rebuild-retry',
  description: 'Retry failed rebuild components C4 (forward model) + C6 (planner) with judge feedback',
  phases: [
    { title: 'Implement', detail: 're-implement fixing the prior judge bugs (opus)' },
    { title: 'Review', detail: 'adversarial judge verifies the bugs are fixed (opus)' },
    { title: 'Measure', detail: 'offline gate via uv: pytest + local 8/8 + push 3/3' },
  ],
}

const REPO = '/Users/ahmed/Documents/ArcAGI3'
const ORDER = [{ id: 'C4', dep: 'forward world model' }, { id: 'C6', dep: 'planner (consumes C4)' }]

const VERDICT = {
  type: 'object',
  properties: {
    pass: { type: 'boolean' },
    prior_bugs_fixed: { type: 'boolean', description: 'are ALL bugs from <Cx>_feedback.md fixed?' },
    issues: { type: 'array', items: { type: 'string' } },
    feedback: { type: 'string' },
  },
  required: ['pass', 'prior_bugs_fixed', 'feedback'],
}
const MEASURE = {
  type: 'object',
  properties: {
    tests_pass: { type: 'boolean' },
    local_levels: { type: 'integer' },
    push_levels: { type: 'integer' },
    summary: { type: 'string' },
  },
  required: ['tests_pass', 'local_levels', 'push_levels', 'summary'],
}

const GUARD = `Use the project venv for ALL verification: \`cd ${REPO} && uv run pytest -q\` and
\`cd ${REPO} && PYTHONPATH=src uv run python -m arcagi3.runner --agent reactive --budget 4000 --quiet\`
(plain \`pytest\`/\`python\` will fail with 'No module named arc_agi' — you MUST use uv run).
NO-REGRESSION IS MANDATORY: pytest green, local 8/8 + 27 levels + push 3/3 unchanged, new code
behind a default-OFF flag. The previous attempt was REVERTED for the judge bugs in the feedback
file — your PRIMARY job is to fix those specific bugs. If you can't, STOP and report.`

const results = []
for (const c of ORDER) {
  phase('Implement')
  const impl = await agent(
    `Re-implement component ${c.id} (${c.dep}) of the ARC-AGI-3 rebuild. The prior attempt was reverted.\nREAD: ${REPO}/rebuild/design/${c.id}.md (blueprint) AND ${REPO}/rebuild/design/${c.id}_feedback.md (the judge's bugs that caused the failure — FIX EVERY ONE) AND the existing committed code (src/arcagi3/{tracking,goals,wm_policy,policy,spatial,perception,movement,world_model,agent}.py).\nImplement the code + offline unit tests, fixing all prior bugs. Match existing style.\n${GUARD}\nReturn exactly what you added/changed and how each prior bug is now fixed.`,
    { label: `impl:${c.id}`, phase: 'Implement', model: 'opus', agentType: 'general-purpose' }
  )
  if (!impl) { results.push({ id: c.id, status: 'impl-failed' }); continue }

  phase('Review')
  const review = await agent(
    `Adversarially review the retry of ${c.id}. Read ${REPO}/rebuild/design/${c.id}_feedback.md (prior bugs) and ${REPO}/rebuild/design/${c.id}.md, then inspect the new code (\`cd ${REPO} && git status\` — new files are untracked; \`git add -N .\` then \`git diff\`). VERIFY every prior bug is actually fixed (set prior_bugs_fixed). Check correctness, that new behaviour is default-OFF, and no push/local regression risk. Be ruthless.`,
    { label: `review:${c.id}`, phase: 'Review', model: 'opus', schema: VERDICT, agentType: 'pr-review-toolkit:code-reviewer' }
  )

  phase('Measure')
  const measure = await agent(
    `Measure no-regression for ${c.id} using the venv:\n  cd ${REPO} && uv run pytest -q\n  cd ${REPO} && PYTHONPATH=src uv run python -m arcagi3.runner --agent reactive --budget 4000 --quiet\nReport tests_pass, local TOTAL levels (expect 27), push levels (expect 3).`,
    { label: `measure:${c.id}`, phase: 'Measure', model: 'haiku', schema: MEASURE, agentType: 'general-purpose' }
  )

  const ok = review && review.pass && review.prior_bugs_fixed && measure &&
             measure.tests_pass && measure.local_levels >= 27 && measure.push_levels >= 3
  if (!ok) {
    await agent(
      `Discard the failed ${c.id} retry: \`cd ${REPO} && git checkout -- src tests 2>/dev/null; git clean -fdq src tests scripts\`. Confirm \`cd ${REPO} && uv run pytest -q\` green and \`git status --short\` clean.`,
      { label: `revert:${c.id}`, phase: 'Measure', model: 'haiku', agentType: 'general-purpose' }
    )
    results.push({ id: c.id, status: 'gate-failed', review, measure })
    log(`${c.id} retry FAILED -> reverted. ${review ? review.feedback : ''}`.slice(0, 300))
    continue
  }
  await agent(
    `Commit the passed ${c.id} retry: \`cd ${REPO} && git add -A && git commit -q -m "feat(rebuild): ${c.id} retry (judge-gated, prior bugs fixed)"\`; show \`git log --oneline -1\`.`,
    { label: `commit:${c.id}`, phase: 'Measure', model: 'haiku', agentType: 'general-purpose' }
  )
  results.push({ id: c.id, status: 'passed', impl, review, measure })
  log(`${c.id} retry PASSED + committed (local ${measure.local_levels}, push ${measure.push_levels}).`)
}

return { results, passed: results.filter((r) => r.status === 'passed').map((r) => r.id) }
