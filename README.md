## MITRE ATT&CK v19 Coverage

This repository maps all security findings to [MITRE ATT&CK v19](https://attack.mitre.org/).

| Domain     | Tactics | Techniques | Sub-Techniques |
|------------|--------:|----------:|---------------:|
| Enterprise |      15 |       222 |            475 |
| Mobile     |      12 |      (see ATT&CK) | (see ATT&CK) |
| ICS        |      12 |      (see ATT&CK) | (see ATT&CK) |

### Export ATT&CK Navigator Layer

```bash
python -m attack_mapping.reporter --output navigator_layer.json
```

Open in [ATT&CK Navigator](https://mitre-attack.github.io/attack-navigator/) to visualize coverage.

### Finding Schema

Every finding object includes:
```json
{
  "attack_mappings": [
    {
      "tactic_id":         "TA0005",
      "tactic_name":       "Defense Evasion",
      "technique_id":      "T1562",
      "technique_name":    "Impair Defenses",
      "subtechnique_id":   "T1562.001",
      "subtechnique_name": "Disable or Modify Tools",
      "domain":            "enterprise",
      "confidence":        0.85,
      "data_sources":      ["..."],
      "platforms":         ["..."],
      "url":               "https://attack.mitre.org/techniques/T1562/001/"
    }
  ]
}
```

### Adversarial ML Lab Specific Mappings

| Finding Type | Techniques |
|--------------|------------|
| adversarial_evasion_success | T1562, T1036.005 |
| adversarial_patch_detected | T1036, T1027 |
| model_bypass_via_perturbation | T1562.001, T1027 |
| transfer_attack_success | T1562, T1190 |
| black_box_query_attack | T1595, T1190 |
| adversarial_robustness_failure | T1562, T1499 |
| certified_defense_bypass | T1562.001 |
| physical_adversarial_attack | T1200, T1036 |