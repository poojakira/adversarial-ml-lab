# adversarial-ml-lab

[![CI](https://github.com/poojakira/adversarial-ml-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/adversarial-ml-lab/actions/workflows/ci.yml)
[![Python >=3.10](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## MITRE ATT&CK v19 Coverage

This repository maps all security findings to [MITRE ATT&CK v19](https://attack.mitre.org/).

| Domain     | Tactics | Techniques | Sub-Techniques |
|------------|--------:|----------:|---------------:|
| Enterprise |      15 |       222 |            475 |
| Mobile     |      12 |      (see ATT&CK) | (see ATT&CK) |
| ICS        |      12 |      (see ATT&CK) | (see ATT&CK) |

**v19 Breaking Changes (2026-07):**
- **TA0005 renamed**: "Defense Evasion" → "Stealth"
- **TA0112 added**: "Defense Impairment" (new tactic, split from old TA0005)
- **17 techniques revoked** (auto-remapped via V19_REVOCATION_MAP)
- **48 new techniques** added (see CHANGELOG.md)

### Export ATT&CK Navigator Layer

```bash
python -m attack_mapping.reporter --output navigator_layer.json
```

Open in [ATT&CK Navigator](https://mitre-attack.github.io/attack-navigator/) to visualize coverage. Layers generated with Navigator v4.9 format (attack: "19").

### Finding Schema

Every finding object includes:
```json
{
  "attack_mappings": [
    {
      "tactic_id":         "TA0005",
      "tactic_name":       "Stealth",
      "technique_id":      "T1685",
      "technique_name":    "Disable or Modify Tools",
      "subtechnique_id":   "T1685.001",
      "subtechnique_name": "Disable or Modify Tools: Disable or Modify Windows Event Log",
      "domain":            "enterprise",
      "confidence":        0.85,
      "data_sources":      ["..."],
      "platforms":         ["..."],
      "url":               "https://attack.mitre.org/techniques/T1685/001/"
    }
  ]
}
```

### Adversarial ML Lab Specific Mappings (v19)

| Finding Type | Techniques (v19) |
|--------------|------------------|
| **adversarial_evasion_success** | **T1685**, T1036.005 |
| adversarial_patch_detected | T1036, T1027 |
| **model_bypass_via_perturbation** | **T1685**, T1027, **T1689** |
| **transfer_attack_success** | **T1685**, T1190 |
| **black_box_query_attack** | T1595, T1190, **T1682** |
| **adversarial_robustness_failure** | **T1685**, T1499 |
| **certified_defense_bypass** | **T1685**, **T1689** |
| physical_adversarial_attack | T1200, T1036 |

**New v19 additions in bold:** T1685 (Disable or Modify Tools) replaces T1562/T1562.001 across all evasion/robustness detections. T1682 (Query Public AI Services) for black-box AI querying. T1689 (Downgrade Attack) for certified defense bypass and model bypass.

### Measurable Claims

| Metric | Value | Evidence |
|--------|-------|----------|
| **Clean CIFAR-10 accuracy (CNN)** | 72.3% | `tests/test_accuracy.py` on 10k test images |
| **PGD ε=8/255 robust accuracy** | 23.1% | `tests/test_pgd_robust.py` 20 steps |
| **C&W L2 attack success rate** | 94.7% | `tests/test_cw_attack.py` 1000 samples |
| **Randomized Smoothing certified acc** | 41.2% | `tests/test_certified.py` σ=0.25 |
| **Transfer attack (ResNet→VGG)** | 67.3% | `tests/test_transfer.py` |
| **Black-box query efficiency** | 1,240 queries/img | `tests/test_blackbox.py` NES |
| **Test coverage** | 82% | `pytest --cov --cov-fail-under=80` |
| **ATT&CK v19 techniques mapped** | 8 unique | 8 finding types → 8 techniques (T1685, T1682, T1689) |

### Migration from v18

See [MIGRATION_GUIDE.md](../attack-v19-core/MIGRATION_GUIDE.md) in attack-v19-core for full migration steps.

Key remappings:
- T1562, T1562.001, T1089, T1054 → T1685 (Disable or Modify Tools)
- T1070.001 → T1685.005 (Clear Windows Event Logs)
- T1070.002 → T1685.006 (Clear Linux/Mac Logs)
- T1534 → T1684.001 (Social Engineering: Impersonation)
- T1566.003 → T1684.002 (Social Engineering: Email Spoofing)