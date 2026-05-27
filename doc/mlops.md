# Machine Learning Operations (MLOps)

> "MLOps is an ML engineering culture and practice that aims at unifying ML system development (Dev) and ML system operation (Ops). Practicing MLOps means that you advocate for automation and monitoring at all steps of ML system construction, including integration, testing, releasing, deployment and infrastructure management."
>
> — Google Cloud Architecture Center

> [!NOTE]
> Unlike machine learning (often defined via Tom M. Mitchell's E/T/P formulation), MLOps has **no single founding author or universal academic definition**. The discipline emerged from industry practice, influenced by DevOps, platform engineering work at companies such as Google, Uber, and Netflix, and Sculley et al.'s 2015 paper, *Hidden Technical Debt in Machine Learning Systems*.


## Levels of MLOps

Google Cloud describes three broad maturity levels for MLOps[^gcp-mlops]:

- **Level 0 - Manual process**: training, validation, deployment, and monitoring are mostly manual.
- **Level 1 - ML pipeline automation**: the training pipeline is automated, enabling continuous training and repeatable model delivery.
- **Level 2 - CI/CD pipeline automation**: both ML pipelines and their surrounding code/infrastructure changes are tested, integrated, and deployed automatically.

[^gcp-mlops]: Google Cloud Architecture Center, *MLOps: Continuous delivery and automation pipelines in machine learning*. https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning