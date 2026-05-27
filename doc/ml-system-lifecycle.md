# ML System Lifecycle

End-to-end stages for building and operating ML systems (roles in `ml-ecosystem.md` map to these stages).

1. **Frame**: define the ML problem, task, success metrics, constraints, assumptions, and product tradeoffs.
2. **Data**: prepare, validate, version, and monitor data quality, labeling, splits, leakage, feature availability, and drift.
3. **Train**: train models reproducibly while understanding model fundamentals, bias-variance tradeoffs, overfitting, regularization, loss functions, optimizers, learning rate schedules, and batch effects.
4. **Evaluate**: analyze model performance with validation/test sets, per-class and slice analysis, thresholding, calibration, robustness checks, and responsible ML considerations.
5. **Package**: export model artifacts, track versions, define preprocessing/postprocessing, and choose appropriate formats and runtimes.
6. **Serve**: run models under real latency, throughput, memory, CPU/GPU placement, scaling, batching, and application integration constraints.
7. **Optimize**: improve inference for target hardware using quantization, distillation, pruning, runtime tuning, and preprocessing/postprocessing optimization.
8. **Test**: build testing and CI/CD workflows for code, data, models, pipelines, performance, and deployment gates.
9. **Operate**: monitor model quality, data drift, latency, errors, cost, and business impact; use alerting, canaries, rollback, retraining, and reliability safeguards.
10. **Communicate**: explain decisions, assumptions, metrics, validation evidence, fairness/privacy/security risks, and product tradeoffs to technical and non-technical stakeholders.