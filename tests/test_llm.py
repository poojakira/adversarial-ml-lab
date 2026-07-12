"""Tests for LLM attack surface (simulated tokenizer/model)."""

from __future__ import annotations

import torch

from adv_lab.attacks.llm import (
    AutoDANAttack,
    GCGAttack,
    SimulatedLLM,
    SimulatedTokenizer,
    embedding_perturbation,
    prompt_injection,
    token_substitution,
    universal_suffix,
)


def _make_llm_setup(num_classes: int = 5, max_length: int = 32):
    """Create a simulated tokenizer and LLM for testing."""
    tokenizer = SimulatedTokenizer(vocab_size=96)
    model = SimulatedLLM(
        vocab_size=96, max_length=max_length, hidden_dim=32, num_classes=num_classes
    )
    model.eval()
    return tokenizer, model


def test_gcg_generates_valid_suffix():
    """GCG attack produces a suffix string and valid logits."""
    tokenizer, model = _make_llm_setup(num_classes=5, max_length=32)
    gcg = GCGAttack(
        model, tokenizer, suffix_length=6, num_candidates=8, steps=5
    )

    suffix_text, logits = gcg.generate_suffix("test prompt", target_class=2, max_length=32)

    assert isinstance(suffix_text, str)
    assert len(suffix_text) > 0
    assert logits.shape == (5,)
    assert torch.isfinite(logits).all()


def test_autodan_produces_valid_jailbreak():
    """AutoDAN evolves a valid jailbreak candidate with a fitness score."""
    tokenizer, model = _make_llm_setup(num_classes=5, max_length=32)
    autodan = AutoDANAttack(
        model, tokenizer, population_size=10, generations=3, mutation_rate=0.2
    )

    prompt_text, score = autodan.generate_jailbreak(
        "hello world", target_class=1, max_length=32
    )

    assert isinstance(prompt_text, str)
    assert isinstance(score, float)


def test_prompt_injection_returns_valid_output():
    """Prompt injection produces valid token IDs and logits."""
    tokenizer, model = _make_llm_setup(num_classes=5, max_length=32)

    token_ids, logits = prompt_injection(
        model, tokenizer, "base prompt", "INJECTED", target_class=0, max_length=32
    )

    assert token_ids.shape == (32,)
    assert logits.shape == (5,)
    assert torch.isfinite(logits).all()


def test_embedding_perturbation_stays_bounded():
    """Embedding perturbation stays within the L2 epsilon ball."""
    tokenizer, model = _make_llm_setup(num_classes=5, max_length=32)
    epsilon = 1.0

    perturbed_emb, logits = embedding_perturbation(
        model, tokenizer, "test input", target_class=3,
        epsilon=epsilon, steps=10, alpha=0.1, max_length=32
    )

    # Check output shapes
    assert perturbed_emb.shape == (32, 32)  # (max_length, hidden_dim)
    assert logits.shape == (5,)

    # Check the perturbation is bounded
    clean_tokens = tokenizer.encode("test input", max_length=32).unsqueeze(0)
    with torch.no_grad():
        clean_emb = model.get_embeddings(clean_tokens)[0]
    delta = perturbed_emb - clean_emb
    l2_norm = delta.view(-1).norm(p=2).item()
    assert l2_norm <= epsilon + 1e-4


def test_token_substitution_modifies_tokens():
    """Token substitution produces a modified text."""
    tokenizer, model = _make_llm_setup(num_classes=5, max_length=32)

    modified_text, logits = token_substitution(
        model, tokenizer, "original text here",
        target_class=4, num_substitutions=3, max_length=32
    )

    assert isinstance(modified_text, str)
    assert logits.shape == (5,)
    # The text should differ from the original (at least some substitutions)
    assert modified_text != "original text here"


def test_universal_suffix_transfers():
    """Universal suffix optimization produces a suffix and per-model scores."""
    tokenizer = SimulatedTokenizer(vocab_size=96)
    # Create multiple models (simulating different families)
    models = []
    for _ in range(2):
        m = SimulatedLLM(vocab_size=96, max_length=32, hidden_dim=32, num_classes=5)
        m.eval()
        models.append(m)

    prompts = ["hello", "test"]

    suffix_text, scores = universal_suffix(
        models, tokenizer, prompts,
        target_class=2, suffix_length=4, steps=3, max_length=32
    )

    assert isinstance(suffix_text, str)
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
