"""
ATT&CK Enricher for adversarial-ml-lab.
"""

from typing import Any

from attack_core.index import ATTACKIndex
from attack_core.mapping import ATTACKMappingBuilder
from attack_core.models import ATTACKMapping


class ATTACKEnricher:
    def __init__(self, index: ATTACKIndex):
        self.index = index
        self.mapping_builder = ATTACKMappingBuilder(index)
        self._rule_table = {
            "adversarial_evasion_success": ["T1685", "T1036.005"],
            "adversarial_patch_detected": ["T1036", "T1027"],
            "model_bypass_via_perturbation": ["T1685", "T1027", "T1689"],
            "transfer_attack_success": ["T1685", "T1190"],
            "black_box_query_attack": ["T1595", "T1190", "T1682"],
            "adversarial_robustness_failure": ["T1685", "T1499"],
            "certified_defense_bypass": ["T1685", "T1689"],
            "physical_adversarial_attack": ["T1200", "T1036"],
        }

    def enrich(self, finding_type: str, metadata: dict[str, Any]) -> list[ATTACKMapping]:
        confidence = metadata.get("confidence", 0.5)
        technique_ids = self._rule_table.get(finding_type, [])
        return self.mapping_builder.build_many(technique_ids, confidence)
