# MITRE ATLAS Mapping

| File | ATLAS Technique | ID | Description |
|------|-----------------|----|-------------|
| attacks/fgsm.py | Craft Adversarial Data | AML.T0043 | Single-step FGSM perturbation |
| attacks/fgsm.py | Evade ML Model | AML.T0015 | Evaluating model evasion |
| attacks/pgd.py | Craft Adversarial Data | AML.T0043 | Iterative PGD perturbation |
| attacks/pgd.py | Evade ML Model | AML.T0015 | Strongest first-order attack |
| attacks/cw.py | Craft Adversarial Data | AML.T0043 | Carlini-Wagner optimization |
| attacks/cw.py | Evade ML Model | AML.T0015 | L2-norm minimization attack |
| defenses/adversarial_training.py | Craft Adversarial Data (defense) | AML.T0043 | Madry AT inner maximization |
| defenses/adversarial_training.py | Backdoor ML Model (defense against) | AML.T0054 | AT reduces backdoor success |
| models/cifar10_resnet18.py | Evade ML Model (target) | AML.T0015 | ResNet-18 attack target |

Reference: https://atlas.mitre.org/
