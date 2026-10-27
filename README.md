# A/B Test Analysis: Lowering the Free-Shipping Threshold ($75 to $50)

End-to-end analysis of a randomized experiment for a fictional online retailer. It tests whether lowering the free-shipping threshold from $75 to $50 raises conversion without hurting profit. The data is synthetic (generated using Claude), generated with a fixed seed so results are reproducible.

**Tools:** Python, pandas, NumPy, SciPy, statsmodels, Matplotlib

## Business question

Should the retailer lower the free-shipping threshold to $50?

**Decision rule:** ship if the conversion lift is significant at the 5% level and contribution margin per user is not significantly lower.

## Experiment design

**Unit of randomization:** User
**Variants:** Control ($75 threshold) vs treatment ($50 threshold)
**Primary metric:** 7-day conversion rate (168 hours from each user's own assignment)
**Secondary metrics:** Revenue per user, average order value (AOV)
**Guardrails:** Margin per user, return rate
**Sample:** 63,960 users (32,011 control, 31,949 treatment)

## Data quality checks

- **Duplicates:** 256 users logged twice in the same variant; kept their earliest row.
- **Contamination:** 40 users appeared in both variants; excluded.
- **Sample ratio mismatch:** none (chi-square p = 0.81).
- **Covariate balance:** device, channel and customer type are similar across variants.
- **Outliers:** 11 users placed orders of $500 or more (8 control, 3 treatment). Results are shown with and without them.

## Results

| Metric | Control | Treatment | Difference | p-value |
|---|---|---|---|---|
| Conversion rate | 3.90% | 4.28% | **+0.38 pp** (95% CI +0.08 to +0.69), +9.8% relative | 0.015 |
| Revenue per user (all) | $3.13 | $3.24 | +$0.11 (CI -0.23 to +0.44) | 0.53 |
| Revenue per user (no bulk orders) | $2.88 | $3.17 | +$0.29 (CI +0.02 to +0.56) | 0.035 |
| Margin per user (all) | $0.99 | $0.90 | -$0.09 (CI -0.20 to +0.02) | 0.096 |
| Margin per user (no bulk orders) | $0.90 | $0.88 | -$0.02 (CI -0.11 to +0.07) | 0.65 |
| AOV (no bulk orders) | $69.85 | $69.79 | flat | n/a |
| Return rate (no bulk orders) | 8.11% | 9.16% | +1.05 pp | 0.32 |

- Conversion rose significantly. The plausible range runs from a small gain (+0.08 pp) to a sizeable one (+0.69 pp).
- Margin did not rise. Extra merchandise margin (+$0.02 per user) was offset by lost shipping revenue (-$0.11 per user). At 1M users, the margin range is about -$199K to +$16K.
- Segments (exploratory): the lift is clearest on mobile (+0.52 pp, Bonferroni p = 0.028). Tests of whether groups differ found no evidence of it (device p = 0.32, customer type p = 0.99), so the effect looks broadly similar across groups.
- Power: the overall test could detect a 0.44 pp lift at 80% power, so the observed +0.38 pp is borderline. Small segments (tablet, desktop) were too small to read.

## Recommendation

By the pre-set rule the test passes: conversion rose significantly and margin was not significantly lower. The margin interval still allows a loss, so I would **ship with monitoring**, or first run a larger **mobile-only follow-up** (about 22,200 users per arm, roughly 24 days) that tracks margin and return rate.

## Limitations

- Only 7 days of outcomes; long-run behavior is not measured.
- Segment results are exploratory and not powered to confirm differences.

## Methods

Chi-square goodness-of-fit (SRM), two-proportion z-test with Wald CI, Welch t-test, Bonferroni correction, Cochran's Q heterogeneity test.

