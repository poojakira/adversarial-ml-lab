# Changelog - adversarial-ml-lab

## [1.0.0] - 2026-07-22

### Changed - ATT&CK v19 Migration

#### Technique Remappings (Revoked -> New)
| Old ID | New ID | Rule Table Keys Affected |
|--------|--------|-------------------------|
| T1562 | T1685 | adversarial_evasion_success, transfer_attack_success, adversarial_robustness_failure |
| T1562.001 | T1685 | model_bypass_via_perturbation, certified_defense_bypass |

#### New Technique Coverage Added
- **T1682** (Query Public AI Services): Added to black_box_query_attack
- **T1689** (Downgrade Attack): Added to model_bypass_via_perturbation, certified_defense_bypass

#### Rule Table Updates
```python
# BEFORE
"adversarial_evasion_success": ["T1562", "T1036.005"],
"model_bypass_via_perturbation": ["T1562.001", "T1027"],
"transfer_attack_success": ["T1562", "T1190"],
"adversarial_robustness_failure": ["T1562", "T1499"],
"certified_defense_bypass": ["T1562.001"],
"black_box_query_attack": ["T1595", "T1190"],

# AFTER
"adversarial_evasion_success": ["T1685", "T1036.005"],
"model_bypass_via_perturbation": ["T1685", "T1027", "T1689"],
"transfer_attack_success": ["T1685", "T1190"],
"adversarial_robustness_failure": ["T1685", "T1499"],
"certified_defense_bypass": ["T1685", "T1689"],
"black_box_query_attack": ["T1595", "T1190", "T1682"],
```

### Added
- T1685 replacing all T1562 references for adversarial evasion/robustness
- T1682 for black-box AI service querying attacks
- T1689 Downgrade Attack for certified defense bypass scenarios

### Migration
See [attack-v19-core MIGRATION_GUIDE.md](../attack-v19-core/MIGRATION_GUIDE.md) for full migration steps.