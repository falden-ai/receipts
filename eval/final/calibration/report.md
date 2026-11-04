# Eval report

Score receipts.eval.score/3, pilot frozen. Bootstrap: 10000 resamples of targets, seed 0. Intervals are 95% (Wilson for rates). A rate over fewer than 10 is marked small n.

## Arms

| arm | attacker | targets | positives | negatives | excluded |
|---|---|---|---|---|---|
| spark | muse-spark-1.3 | 150 | 90 | 60 | 0 |
| glimmer | muse-glimmer-30b | 150 | 90 | 60 | 0 |

## Rates, REG-561 (blockers only)

| arm | WDR | false alarms | CAR | precision | precision bootstrap | MCC | MCC bootstrap | invalid repros |
|---|---|---|---|---|---|---|---|---|
| spark | 35.6% [26.4%, 45.8%] (32/90) | 40.0% [28.6%, 52.6%] (24/60) | 60.0% [47.4%, 71.4%] (36/60) | 65.2% [53.1%, 75.5%] (43/66) | [0.534, 0.763] (10000 draws) | 0.066 | [-0.095, 0.224] (10000 draws) | 39.6% [33.0%, 46.6%] (78/197) |
| glimmer | 13.3% [7.8%, 21.9%] (12/90) | 20.0% [11.8%, 31.8%] (12/60) | 80.0% [68.2%, 88.2%] (48/60) | 50.0% [33.2%, 66.8%] (15/30) | [0.320, 0.680] (10000 draws) | 0.000 | [-0.161, 0.160] (10000 draws) | 48.6% [33.0%, 64.4%] (17/35) |

## Rates with fix-validity catches (REG-577)

| arm | WDR | false alarms | CAR | precision | precision bootstrap | MCC | MCC bootstrap | invalid repros |
|---|---|---|---|---|---|---|---|---|
| spark | 54.4% [44.2%, 64.3%] (49/90) | 40.0% [28.6%, 52.6%] (24/60) | 60.0% [47.4%, 71.4%] (36/60) | 74.1% [63.6%, 82.4%] (60/81) | [0.644, 0.833] (10000 draws) | 0.229 | [0.071, 0.386] (10000 draws) | 56.5% [51.6%, 61.3%] (221/391) |
| glimmer | 18.9% [12.1%, 28.2%] (17/90) | 20.0% [11.8%, 31.8%] (12/60) | 80.0% [68.2%, 88.2%] (48/60) | 57.1% [40.9%, 72.0%] (20/35) | [0.406, 0.737] (10000 draws) | 0.064 | [-0.094, 0.219] (10000 draws) | 79.4% [71.9%, 85.4%] (108/136) |

### Targets to read, REG-561

- spark: GOOD label disputed: 20260217_mini-v2.0.0_claude-4-6-opus~sphinx-doc__sphinx-7454, 20260217_mini-v2.0.0_claude-4-6-opus~pydata__xarray-4687, 20260217_mini-v2.0.0_gemini-3-flash-high~django__django-13590, 20260217_mini-v2.0.0_glm-5-high~django__django-16950, 20260217_mini-v2.0.0_glm-5-high~django__django-13590, 20260217_mini-v2.0.0_gemini-3-flash-high~django__django-11964, 20260217_mini-v2.0.0_glm-5-high~django__django-12125, 20260217_mini-v2.0.0_glm-5-high~matplotlib__matplotlib-22871, 20260217_mini-v2.0.0_gemini-3-flash-high~sympy__sympy-14711, 20250807_openhands_gpt5~sympy__sympy-12096, 20250807_openhands_gpt5~pydata__xarray-6461
- glimmer: GOOD label disputed: 20260217_mini-v2.0.0_claude-4-6-opus~sphinx-doc__sphinx-7454, 20260217_mini-v2.0.0_gemini-3-flash-high~django__django-13315, 20260217_mini-v2.0.0_gemini-3-flash-high~django__django-11964

### Targets to read, with fix-validity

- spark: GOOD label disputed: 20260217_mini-v2.0.0_claude-4-6-opus~sphinx-doc__sphinx-7454, 20260217_mini-v2.0.0_claude-4-6-opus~pydata__xarray-4687, 20260217_mini-v2.0.0_gemini-3-flash-high~django__django-13590, 20260217_mini-v2.0.0_glm-5-high~django__django-16950, 20260217_mini-v2.0.0_glm-5-high~django__django-13590, 20260217_mini-v2.0.0_gemini-3-flash-high~django__django-11964, 20260217_mini-v2.0.0_glm-5-high~django__django-12125, 20260217_mini-v2.0.0_glm-5-high~matplotlib__matplotlib-22871, 20260217_mini-v2.0.0_gemini-3-flash-high~sympy__sympy-14711, 20250807_openhands_gpt5~sympy__sympy-12096, 20250807_openhands_gpt5~pydata__xarray-6461
- glimmer: GOOD label disputed: 20260217_mini-v2.0.0_claude-4-6-opus~sphinx-doc__sphinx-7454, 20260217_mini-v2.0.0_gemini-3-flash-high~django__django-13315, 20260217_mini-v2.0.0_gemini-3-flash-high~django__django-11964

## PPV from WDR and CAR

| arm | variant | sensitivity (WDR) | specificity (CAR) | PPV at 50% | PPV at 20% | PPV at 2.66% |
|---|---|---|---|---|---|---|
| spark | REG-561 | 35.6% [26.4%, 45.8%] (32/90) | 60.0% [47.4%, 71.4%] (36/60) | 47.1% | 18.2% | 2.4% |
| spark | with fv | 54.4% [44.2%, 64.3%] (49/90) | 60.0% [47.4%, 71.4%] (36/60) | 57.6% | 25.4% | 3.6% |
| glimmer | REG-561 | 13.3% [7.8%, 21.9%] (12/90) | 80.0% [68.2%, 88.2%] (48/60) | 40.0% | 14.3% | 1.8% |
| glimmer | with fv | 18.9% [12.1%, 28.2%] (17/90) | 80.0% [68.2%, 88.2%] (48/60) | 48.6% | 19.1% | 2.5% |

## Cost by stage

| arm | at2_runs | at2_stage1 | at3_intent | at3_suite | attack.correctness.explore | attack.correctness.sandbox | attack.correctness.write | refuter | skeptic | surrogate | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spark | $0.00 | $25.811538 | $0.139463 | $0.00 | $36.366558 | $0.00 | $4.765205 | $20.187675 | $0.00 | $43.327804 | $130.598243 |
| glimmer | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $4.36668 | $0.00 | $3.753566 | $8.120246 |

## Efficiency

| arm | cost per catch | cost per catch with fv | cost per target p50 | p95 | time to catch p50 | p95 |
|---|---|---|---|---|---|---|
| spark | $3.037168 (43 caught) | $2.176637 (60 caught) | $0.869104 | $1.351768 | 867 s | 1271 s (43 timed) |
| glimmer | $0.54135 (15 caught) | $0.406012 (20 caught) | $0.00 | $0.360849 | 399 s | 742 s (15 timed) |

## McNemar exact, between arms

| arms | positives, caught | negatives, alarm |
|---|---|---|
| spark vs glimmer | p 0.0000 (20 vs 0 discordant of 90) | p 0.0075 (15 vs 3 discordant of 60) |

## Refutations (SK2), suspected findings by category

| arm | intended_change | out_of_scope | test_bug | unsupported_by_quote |
|---|---|---|---|---|
| spark | 104 | 2 | 18 | 3 |
| glimmer | 41 | 0 | 2 | 1 |
