# Model selection

## The question
The model predicts whether a customer will churn. Missed churners may not receive retention support; false alarms can waste retention resources.

## Candidates
All candidates used the same data and stratified 80/20 split with random_state=42.

| Candidate | Why tried | AUC | F1 | Lowest slice F1 | p95 ms | Size KB | Run ID |
|---|---|---|---|---|---|---|---|
| logreg_c05 | Simple regularized baseline, C=0.5 | 0.7846 | 0.5042 | 0.0000 | 9.5537 | 3.5479 | 9349da231feb4b8f8a3f97fac604ab26 |
| forest_depth6 | Nonlinear interactions; 100 trees, depth 6 | 0.7666 | 0.4483 | 0.0000 | 17.3441 | 751.6895 | 796fe4633f274d138ad86aa931a3db72 |
| boosting_depth2 | Sequential shallow trees; 80 trees, depth 2 | 0.7743 | 0.4628 | 0.0000 | 6.9219 | 72.1289 | 6016bac8924f4281871e72be6fd5a38d |

## Contract slices
| Candidate | Month-to-month F1 | One-year F1 | Two-year F1 |
|---|---|---|---|
| logreg_c05 | 0.6237 | 0.1250 | 0.0000 |
| forest_depth6 | 0.5618 | 0.1250 | 0.0000 |
| boosting_depth2 | 0.5745 | 0.1250 | 0.0000 |

## The champion
I selected logreg_c05 for the highest overall F1 and AUC, the highest month-to-month F1, and the smallest model.
I accepted approximately 2.63 ms more p95 latency than boosting.
All candidates tie on one-year and two-year F1, so this choice does not improve those groups.

## My promotion rules
- Latency budget: p95 <= 50 ms. This is a lab target with headroom above the observed timings for variation on CI runners.
- Minimum gain: 0.01 absolute F1, equivalent to one percentage point. This avoids replacing the champion for tiny improvements; it is not a statistical significance test.
- Only successfully completed runs with a logged model are eligible.

## What I would check next
Check sample counts, positive churn counts, and confusion matrices for each contract group.
Two-year F1 is zero for every candidate; these results alone do not explain why.
Assess threshold choice and performance on fresh data before real production use.

Data fingerprint recorded by the starter: 8685886ae7ce (12-character SHA-256 prefix).
