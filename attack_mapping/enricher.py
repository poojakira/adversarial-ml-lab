"""
ATT&CK Enricher for adversarial-ml-lab.
"""
from attack_core.index import ATTACKIndex
from attack_core.models import ATTACKMapping
from typing import List, Dict, Any


class ATTACKEnricher:
    def __init__(self, index: ATTACKIndex):
        self.index = index
        self._rule_table = {
            "adversarial_evasion_success": ["T1562", "T1036.005"],
            "adversarial_patch_detected": ["T1036", "T1027"],
            "model_bypass_via_perturbation": ["T1562.001", "T1027"],
            "transfer_attack_success": ["T1562", "T1190"],
            "black_box_query_attack": ["T1595", "T1190"],
            "adversarial_robustness_failure": ["T1562", "T1499"],
            "certified_defense_bypass": ["T1562.001"],
            "physical_adversarial_attack": ["T1200", "T1036"],
        }

    def enrich(self, finding_type: str, metadata: Dict[str, Any]) -> List[ATTACKMapping]:
        technique_ids = self._rule_table.get(finding_type, [])
        mappings = []
        for tid in technique_ids:
            tech = self.index.get(tid)
            if tech:
                tactic = self.index._tactics.get(tech.tactic_ids[0] if tech.tactic_ids else "", None)
                mappings.append(ATTACKMapping(
                    tactic_id=tech.tactic_ids[0] if tech.tactic_ids else "unknown",
                    tactic_name=tactic.name if tactic else "unknown",
                    technique_id=tech.attack_id,
                    technique_name=tech.name,
                    subtechnique_id=tech.attack_id if tech.is_subtechnique else None,
                    subtechnique_name=tech.name if tech.is_subtechnique else None,
                    domain=tech.domain,
                    confidence=metadata.get("confidence", 0.5),
                    data_sources=tech.data_sources,
                    platforms=tech.platforms,
                    url=tech.url,
                ))
        return mappings