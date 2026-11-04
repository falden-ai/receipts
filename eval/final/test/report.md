# Eval report

Score receipts.eval.score/3, pilot frozen. Bootstrap: 10000 resamples of targets, seed 0. Intervals are 95% (Wilson for rates). A rate over fewer than 10 is marked small n.

## Arms

| arm | attacker | targets | positives | negatives | excluded |
|---|---|---|---|---|---|
| spark | muse-spark-1.3 | 225 | 135 | 90 | 0 |
| glimmer | muse-glimmer-30b | 225 | 135 | 90 | 0 |

## Rates, REG-561 (blockers only)

| arm | WDR | false alarms | CAR | precision | precision bootstrap | MCC | MCC bootstrap | invalid repros |
|---|---|---|---|---|---|---|---|---|
| spark | 41.5% [33.5%, 49.9%] (56/135) | 36.7% [27.4%, 47.0%] (33/90) | 63.3% [53.0%, 72.6%] (57/90) | 61.1% [51.8%, 69.5%] (69/113) | [0.521, 0.699] (10000 draws) | 0.221 | [0.094, 0.347] (10000 draws) | 43.0% [37.4%, 48.7%] (125/291) |
| glimmer | 11.1% [6.9%, 17.5%] (15/135) | 12.2% [7.0%, 20.6%] (11/90) | 87.8% [79.4%, 93.0%] (79/90) | 55.9% [39.5%, 71.1%] (19/34) | [0.393, 0.727] (10000 draws) | 0.066 | [-0.062, 0.187] (10000 draws) | 44.2% [31.6%, 57.7%] (23/52) |

## Rates with fix-validity catches (REG-577)

| arm | WDR | false alarms | CAR | precision | precision bootstrap | MCC | MCC bootstrap | invalid repros |
|---|---|---|---|---|---|---|---|---|
| spark | 55.6% [47.1%, 63.7%] (75/135) | 38.9% [29.5%, 49.2%] (35/90) | 61.1% [50.8%, 70.5%] (55/90) | 71.7% [63.3%, 78.8%] (91/127) | [0.636, 0.794] (10000 draws) | 0.289 | [0.162, 0.413] (10000 draws) | 59.4% [55.4%, 63.2%] (360/606) |
| glimmer | 17.0% [11.6%, 24.3%] (23/135) | 12.2% [7.0%, 20.6%] (11/90) | 87.8% [79.4%, 93.0%] (79/90) | 64.3% [49.2%, 77.0%] (27/42) | [0.500, 0.786] (10000 draws) | 0.135 | [0.010, 0.250] (10000 draws) | 75.6% [68.6%, 81.5%] (127/168) |

### Targets to read, REG-561

- spark: GOOD label disputed: 20260217_mini-v2.0.0_claude-4-6-opus~django__django-13297, 20250807_openhands_gpt5~pytest-dev__pytest-7324, 20260217_mini-v2.0.0_claude-4-6-opus~django__django-15563, 20250807_openhands_gpt5~psf__requests-1142, 20260217_mini-v2.0.0_gemini-3-flash-high~astropy__astropy-13579, 20260217_mini-v2.0.0_claude-4-6-opus~django__django-16819, 20250807_openhands_gpt5~django__django-14017, 20260217_mini-v2.0.0_minimax-2-5-high~pylint-dev__pylint-7080, 20260217_mini-v2.0.0_gemini-3-flash-high~pydata__xarray-3677, 20260217_mini-v2.0.0_minimax-2-5-high~sympy__sympy-19346, 20260217_mini-v2.0.0_glm-5-high~sphinx-doc__sphinx-7889, 20260217_mini-v2.0.0_gemini-3-flash-high~pylint-dev__pylint-7080, 20260217_mini-v2.0.0_gemini-3-flash-high~matplotlib__matplotlib-23314
- glimmer: GOOD label disputed: 20250807_openhands_gpt5~pytest-dev__pytest-7324, 20250807_openhands_gpt5~psf__requests-1142, 20250807_openhands_gpt5~django__django-14017, 20260217_mini-v2.0.0_gemini-3-flash-high~pydata__xarray-3677

### Targets to read, with fix-validity

- spark: GOOD label disputed: 20260217_mini-v2.0.0_claude-4-6-opus~django__django-13297, 20260217_mini-v2.0.0_minimax-2-5-high~sympy__sympy-13551, 20250807_openhands_gpt5~pytest-dev__pytest-7324, 20260217_mini-v2.0.0_claude-4-6-opus~django__django-15563, 20260217_mini-v2.0.0_gemini-3-flash-high~django__django-11999, 20250807_openhands_gpt5~psf__requests-1142, 20260217_mini-v2.0.0_gemini-3-flash-high~astropy__astropy-13579, 20260217_mini-v2.0.0_glm-5-high~sympy__sympy-24443, 20260217_mini-v2.0.0_claude-4-6-opus~django__django-16819, 20250807_openhands_gpt5~django__django-14017, 20260217_mini-v2.0.0_minimax-2-5-high~pylint-dev__pylint-7080, 20260217_mini-v2.0.0_gemini-3-flash-high~pydata__xarray-3677, 20260217_mini-v2.0.0_minimax-2-5-high~sympy__sympy-19346, 20260217_mini-v2.0.0_glm-5-high~sphinx-doc__sphinx-7889, 20260217_mini-v2.0.0_gemini-3-flash-high~pylint-dev__pylint-7080, 20260217_mini-v2.0.0_gemini-3-flash-high~matplotlib__matplotlib-23314
- glimmer: GOOD label disputed: 20250807_openhands_gpt5~pytest-dev__pytest-7324, 20250807_openhands_gpt5~psf__requests-1142, 20250807_openhands_gpt5~django__django-14017, 20260217_mini-v2.0.0_gemini-3-flash-high~pydata__xarray-3677

## PPV from WDR and CAR

| arm | variant | sensitivity (WDR) | specificity (CAR) | PPV at 50% | PPV at 20% | PPV at 2.66% |
|---|---|---|---|---|---|---|
| spark | REG-561 | 41.5% [33.5%, 49.9%] (56/135) | 63.3% [53.0%, 72.6%] (57/90) | 53.1% | 22.0% | 3.0% |
| spark | with fv | 55.6% [47.1%, 63.7%] (75/135) | 61.1% [50.8%, 70.5%] (55/90) | 58.8% | 26.3% | 3.8% |
| glimmer | REG-561 | 11.1% [6.9%, 17.5%] (15/135) | 87.8% [79.4%, 93.0%] (79/90) | 47.6% | 18.5% | 2.4% |
| glimmer | with fv | 17.0% [11.6%, 24.3%] (23/135) | 87.8% [79.4%, 93.0%] (79/90) | 58.2% | 25.8% | 3.7% |

## Cost by stage

| arm | at2_runs | at2_stage1 | at3_intent | at3_suite | attack.correctness.explore | attack.correctness.sandbox | attack.correctness.write | refuter | skeptic | surrogate | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spark | $0.00 | $38.991989 | $0.154671 | $0.00 | $55.848235 | $0.00 | $6.497459 | $37.023608 | $0.00 | $70.379404 | $208.895366 |
| glimmer | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $9.403017 | $0.00 | $4.498738 | $13.901755 |

## Efficiency

| arm | cost per catch | cost per catch with fv | cost per target p50 | p95 | time to catch p50 | p95 |
|---|---|---|---|---|---|---|
| spark | $3.027469 (69 caught) | $2.295553 (91 caught) | $0.901049 | $1.447147 | 801 s | 1367 s (69 timed) |
| glimmer | $0.731671 (19 caught) | $0.51488 (27 caught) | $0.00 | $0.349771 | 328 s | 1144 s (19 timed) |

## McNemar exact, between arms

| arms | positives, caught | negatives, alarm |
|---|---|---|
| spark vs glimmer | p 0.0000 (41 vs 0 discordant of 135) | p 0.0000 (23 vs 1 discordant of 90) |

## Refutations (SK2), suspected findings by category

| arm | intended_change | out_of_scope | test_bug | unsupported_by_quote |
|---|---|---|---|---|
| spark | 222 | 7 | 48 | 2 |
| glimmer | 97 | 3 | 11 | 0 |
