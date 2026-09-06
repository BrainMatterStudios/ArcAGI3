# field-refresh — working notes (2026-09-06 ~15:10 UTC)

Artifacts in this directory:
- lb_20260906.json (Kaggle API top-20), lbcsv/*.csv (full public LB incl. TeamMemberUserNames, 2,837 teams)
- lb_history_0906.json (monitor endpoint https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run; running maxima + runtime minutes), team_histories.txt
- kernels_daterun.txt, kernels_votes.txt (kaggle kernels list), kernels/<ref>/ (pulled notebooks + metadata), diffs/<ref>.diff (cell diffs vs keith V14)
- bundles/taaf-avo-v27-bundle, bundles/taaf-polyphony-v25-bundle (raist321 re-hosts of jakobbrggen/taaf-kaggle-source v27/v25), bundles/duck-qwen38-nvfp4-bundle-v1
- sonpham_docs/ (astra README + trace audit, prolong README, etc.), github_search*.txt, team_artifacts*.txt, kyutai_meta/
- forum snapshots under repo .playwright-mcp/page-2026-09-06T15-0*.yml

Key numbers:
- V14-bytes per-draw cohort n=9 {2.80,3.22,3.38,3.25(us),3.40,2.62,4.33,2.96,4.03}: mean 3.33 sd 0.55; expected max of 50 draws 4.58 (90% 4.20-5.04)
- Daily >=0.8 jumps: 09-01 21, 09-02 54, 09-03 62, 09-04 54, 09-05 54; from<=2.2 landing medians 2.75/2.96/3.04/2.89/3.04
- 5+ entrants: Third Intelligence 1.80->3.97->5.53->6.43 (09-02/03/04); Fususu 1.81->3.20->3.59->5.43 (09-01/02/03); Nader 3.19->3.67->5.05 (08-29/31, 09-04); jinbo 2.01->5.49 (09-02); Kyutai 3.37->4.75->4.90 (09-04/05); Shuhan Yang 3.79->4.53->4.74 (3 subs, 09-02/03/04); Franzen 4.05->4.49->6.66->7.63 (09-03/04/05)
- None of them has any public kernel/dataset/model touching ARC-3 since 09-01 (Kyutai: romainfabre vllm nightly wheelhouse 08-17 only).
- Rig: one 25-game arm ~2.7-3.0 h RTX PRO 6000 ~$8-9; base 36 levels (1.44/game); step = >=48 levels; per-game paired sd ~1.2 lv.
