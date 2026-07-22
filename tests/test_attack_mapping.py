import pytest
from attack_core import ATTACKLoader, ATTACKIndex
from attack_mapping.enricher import ATTACKEnricher


@pytest.fixture
def enricher():
    loader = ATTACKLoader()
    index = ATTACKIndex(loader)
    return ATTACKEnricher(index)


class TestAdversarialMLEnricher:
    def test_adversarial_evasion(self, enricher):
        mappings = enricher.enrich("adversarial_evasion_success", {"confidence": 0.9})
        technique_ids = [m.technique_id for m in mappings]
        assert "T1562" in technique_ids
        assert "T1036.005" in technique_ids

    def test_adversarial_patch(self, enricher):
        mappings = enricher.enrich("adversarial_patch_detected", {"confidence": 0.85})
        technique_ids = [m.technique_id for m in mappings]
        assert "T1036" in technique_ids
        assert "T1027" in technique_ids

    def test_physical_attack(self, enricher):
        mappings = enricher.enrich("physical_adversarial_attack", {"confidence": 0.95})
        technique_ids = [m.technique_id for m in mappings]
        assert "T1200" in technique_ids
        assert "T1036" in technique_ids