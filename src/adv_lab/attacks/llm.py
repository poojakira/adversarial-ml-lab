"""LLM adversarial attack surface: attacks against language model inference.

Since real LLM weights cannot be downloaded in this environment, all attacks
use a **simulated tokenizer** (character-level mapping to indices) and a
**simulated LLM** (small feedforward network over one-hot encoded token
sequences). In production, replace ``SimulatedTokenizer`` and ``SimulatedLLM``
with real model/tokenizer instances.

Attacks implemented:

* **GCG** -- Zou et al., "Universal and Transferable Adversarial Attacks on
  Aligned Language Models" (2023). Greedy Coordinate Gradient suffix generation.
* **AutoDAN** -- Liu et al., "AutoDAN: Generating Stealthy Jailbreak Prompts on
  Aligned Large Language Models" (2024). Genetic algorithm for jailbreak prompts.
* **Prompt Injection** -- Perez & Ribeiro, "Ignore This Title and HackAPrompt"
  (2023). Context poisoning via adversarial prefix/suffix.
* **Embedding Perturbation** -- Continuous perturbation in embedding space,
  analogous to FGSM/PGD but in the token embedding manifold.
* **Token Substitution** -- Gradient-guided discrete token replacement,
  following Ebrahimi et al., "HotFlip: White-Box Adversarial Examples for
  Text Classification" (ACL 2018).
* **Universal Suffix** -- A fixed adversarial suffix that transfers across
  model instances, based on Zou et al. (2023) universality experiments.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# --------------------------------------------------------------------------- #
# Simulated Tokenizer and LLM
# --------------------------------------------------------------------------- #


class SimulatedTokenizer:
    """Character-level tokenizer for demonstration purposes.

    Maps each printable ASCII character to a unique index. In production,
    replace with a real BPE/SentencePiece tokenizer.

    Attributes:
        vocab_size: number of unique tokens (printable ASCII subset).
        pad_token_id: index used for padding.
    """

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        # Character to index mapping (ASCII printable range)
        self._char_to_idx: dict[str, int] = {}
        self._idx_to_char: dict[int, str] = {}
        for i in range(1, min(vocab_size, 127)):
            ch = chr(i + 31)  # Start from space (32) mapped to index 1
            self._char_to_idx[ch] = i
            self._idx_to_char[i] = ch
        self._idx_to_char[0] = "<pad>"

    def encode(self, text: str, max_length: int = 64) -> Tensor:
        """Encode text to token indices.

        Args:
            text: input string.
            max_length: maximum sequence length (pad or truncate).

        Returns:
            Long tensor of shape ``(max_length,)`` with token indices.
        """
        indices = []
        for ch in text[:max_length]:
            idx = self._char_to_idx.get(ch, self.pad_token_id)
            indices.append(idx)
        # Pad to max_length
        while len(indices) < max_length:
            indices.append(self.pad_token_id)
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, token_ids: Tensor) -> str:
        """Decode token indices back to text.

        Args:
            token_ids: tensor of token indices.

        Returns:
            Decoded string.
        """
        chars = []
        for idx in token_ids.tolist():
            ch = self._idx_to_char.get(int(idx), "?")
            if ch != "<pad>":
                chars.append(ch)
        return "".join(chars)

    def batch_encode(self, texts: list[str], max_length: int = 64) -> Tensor:
        """Encode a batch of texts.

        Returns:
            Long tensor of shape ``(batch_size, max_length)``.
        """
        return torch.stack([self.encode(t, max_length) for t in texts])


class SimulatedLLM(nn.Module):
    """Small feedforward network simulating an LLM for attack development.

    Architecture: embedding -> flatten -> linear -> ReLU -> linear -> logits.
    This produces next-token logits given a sequence of token indices. In
    production, replace with a real transformer-based language model.

    Args:
        vocab_size: number of unique tokens.
        max_length: maximum input sequence length.
        hidden_dim: hidden layer dimension.
        num_classes: number of output classes (for classification tasks) or
            vocab_size (for next-token prediction).
    """

    def __init__(
        self,
        vocab_size: int = 128,
        max_length: int = 64,
        hidden_dim: int = 64,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        if num_classes is None:
            num_classes = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(max_length * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """Forward pass.

        Args:
            token_ids: ``(N, seq_len)`` long tensor of token indices.

        Returns:
            Logits of shape ``(N, num_classes)``.
        """
        emb = self.embedding(token_ids)  # (N, seq_len, hidden_dim)
        flat = emb.view(emb.shape[0], -1)  # (N, seq_len * hidden_dim)
        return self.classifier(flat)

    def get_embeddings(self, token_ids: Tensor) -> Tensor:
        """Get continuous embeddings for token_ids.

        Args:
            token_ids: ``(N, seq_len)`` long tensor.

        Returns:
            Embeddings of shape ``(N, seq_len, hidden_dim)``.
        """
        return self.embedding(token_ids)

    def forward_from_embeddings(self, embeddings: Tensor) -> Tensor:
        """Forward pass from continuous embeddings (bypasses discrete tokens).

        Args:
            embeddings: ``(N, seq_len, hidden_dim)`` float tensor.

        Returns:
            Logits of shape ``(N, num_classes)``.
        """
        flat = embeddings.view(embeddings.shape[0], -1)
        return self.classifier(flat)


# --------------------------------------------------------------------------- #
# GCG Attack
# --------------------------------------------------------------------------- #


class GCGAttack:
    """Greedy Coordinate Gradient (GCG) adversarial suffix generation.

    Zou et al., "Universal and Transferable Adversarial Attacks on Aligned
    Language Models" (2023).

    The attack appends a suffix to the input prompt and optimizes it
    token-by-token using gradient information. At each position, it computes
    the gradient of the loss with respect to the one-hot token representation,
    then greedily selects the token that maximizes the attack objective.

    Args:
        model: a ``SimulatedLLM`` instance (or compatible model).
        tokenizer: a ``SimulatedTokenizer`` instance.
        suffix_length: number of suffix tokens to optimize.
        num_candidates: number of candidate replacements to evaluate per position.
        steps: number of optimization iterations.
    """

    def __init__(
        self,
        model: SimulatedLLM,
        tokenizer: SimulatedTokenizer,
        suffix_length: int = 10,
        num_candidates: int = 16,
        steps: int = 20,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.suffix_length = suffix_length
        self.num_candidates = num_candidates
        self.steps = steps

    def generate_suffix(
        self,
        prompt: str,
        target_class: int,
        max_length: int = 64,
    ) -> tuple[str, Tensor]:
        """Generate an adversarial suffix for the given prompt.

        Optimizes a suffix appended to ``prompt`` so that the model predicts
        ``target_class`` (targeted attack).

        Args:
            prompt: the base input prompt.
            target_class: desired model output class.
            max_length: maximum total sequence length.

        Returns:
            Tuple of ``(suffix_text, final_logits)`` where suffix_text is the
            decoded adversarial suffix and final_logits are the model outputs.
        """
        self.model.eval()

        # Encode the prompt
        prompt_tokens = self.tokenizer.encode(prompt, max_length=max_length)
        prompt_len = min(len(prompt), max_length - self.suffix_length)

        # Initialize suffix with random tokens
        suffix_tokens = torch.randint(
            1, self.tokenizer.vocab_size, (self.suffix_length,)
        )

        # Full sequence: prompt + suffix (truncated/padded to max_length)
        full_seq = prompt_tokens.clone()
        suffix_start = prompt_len

        for step in range(self.steps):
            # Place current suffix into the sequence
            full_seq_current = full_seq.clone()
            end_pos = min(suffix_start + self.suffix_length, max_length)
            actual_suffix_len = end_pos - suffix_start
            full_seq_current[suffix_start:end_pos] = suffix_tokens[:actual_suffix_len]

            # Compute gradient w.r.t. one-hot token representation
            token_input = full_seq_current.unsqueeze(0)  # (1, max_length)

            # Use embedding to get gradients
            emb = self.model.embedding(token_input)  # (1, seq, hidden)
            emb_param = emb.clone().detach().requires_grad_(True)
            logits = self.model.forward_from_embeddings(emb_param)
            target = torch.tensor([target_class], dtype=torch.long)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()

            grad = emb_param.grad[0]  # (seq, hidden)

            # For each suffix position, find the best replacement token
            for pos_offset in range(actual_suffix_len):
                pos = suffix_start + pos_offset
                if pos >= max_length:
                    break

                # Gradient at this position
                pos_grad = grad[pos]  # (hidden,)

                # Score each candidate token by its alignment with -gradient
                # (we want to minimize loss, so move against the gradient)
                candidates = torch.randint(
                    1, self.tokenizer.vocab_size, (self.num_candidates,)
                )
                best_token = suffix_tokens[pos_offset]
                best_score = float("inf")

                for cand in candidates:
                    cand_emb = self.model.embedding(cand.unsqueeze(0))[0]
                    # Score: dot product with gradient (lower is better for
                    # minimizing CE loss toward target)
                    score = (cand_emb * pos_grad).sum().item()
                    if score < best_score:
                        best_score = score
                        best_token = cand.item()

                suffix_tokens[pos_offset] = best_token

        # Final evaluation
        full_seq_final = full_seq.clone()
        end_pos = min(suffix_start + self.suffix_length, max_length)
        actual_suffix_len = end_pos - suffix_start
        full_seq_final[suffix_start:end_pos] = suffix_tokens[:actual_suffix_len]

        with torch.no_grad():
            final_logits = self.model(full_seq_final.unsqueeze(0))

        suffix_text = self.tokenizer.decode(suffix_tokens[:actual_suffix_len])
        return suffix_text, final_logits[0]


# --------------------------------------------------------------------------- #
# AutoDAN Attack
# --------------------------------------------------------------------------- #


class AutoDANAttack:
    """AutoDAN: Genetic algorithm for jailbreak prompt construction.

    Liu et al., "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned
    Large Language Models" (2024).

    Evolves a population of jailbreak candidates using crossover and mutation,
    selecting for prompts that elicit the target behavior from the model while
    maintaining fluency.

    Args:
        model: a ``SimulatedLLM`` instance.
        tokenizer: a ``SimulatedTokenizer`` instance.
        population_size: number of candidates in each generation.
        generations: number of evolutionary generations.
        mutation_rate: probability of mutating each token.
        crossover_rate: probability of crossover between parents.
    """

    def __init__(
        self,
        model: SimulatedLLM,
        tokenizer: SimulatedTokenizer,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def generate_jailbreak(
        self,
        base_prompt: str,
        target_class: int,
        max_length: int = 64,
    ) -> tuple[str, float]:
        """Evolve a jailbreak prompt via genetic algorithm.

        Starting from random variations of ``base_prompt``, evolves candidates
        to maximize the probability of the target class.

        Args:
            base_prompt: seed prompt to evolve from.
            target_class: desired model output class.
            max_length: maximum sequence length.

        Returns:
            Tuple of ``(best_prompt_text, best_score)`` where score is the
            target class logit of the best candidate.
        """
        self.model.eval()

        # Initialize population with mutations of the base prompt
        population = []
        base_tokens = self.tokenizer.encode(base_prompt, max_length=max_length)
        for _ in range(self.population_size):
            candidate = base_tokens.clone()
            # Apply random mutations
            for pos in range(max_length):
                if random.random() < self.mutation_rate * 3:  # Higher initial mutation
                    candidate[pos] = random.randint(1, self.tokenizer.vocab_size - 1)
            population.append(candidate)

        best_candidate = base_tokens.clone()
        best_score = float("-inf")

        for gen in range(self.generations):
            # Evaluate fitness of each candidate
            fitness_scores = []
            batch = torch.stack(population)
            with torch.no_grad():
                logits = self.model(batch)
                target_logits = logits[:, target_class]

            for idx in range(len(population)):
                score = target_logits[idx].item()
                fitness_scores.append(score)
                if score > best_score:
                    best_score = score
                    best_candidate = population[idx].clone()

            # Selection: tournament selection
            new_population = []
            for _ in range(self.population_size):
                # Tournament of 3
                contestants = random.sample(range(len(population)), min(3, len(population)))
                winner = max(contestants, key=lambda i: fitness_scores[i])
                new_population.append(population[winner].clone())

            # Crossover
            for i in range(0, len(new_population) - 1, 2):
                if random.random() < self.crossover_rate:
                    crossover_point = random.randint(1, max_length - 1)
                    child1 = torch.cat([
                        new_population[i][:crossover_point],
                        new_population[i + 1][crossover_point:],
                    ])
                    child2 = torch.cat([
                        new_population[i + 1][:crossover_point],
                        new_population[i][crossover_point:],
                    ])
                    new_population[i] = child1
                    new_population[i + 1] = child2

            # Mutation
            for i in range(len(new_population)):
                for pos in range(max_length):
                    if random.random() < self.mutation_rate:
                        new_population[i][pos] = random.randint(
                            1, self.tokenizer.vocab_size - 1
                        )

            population = new_population

        return self.tokenizer.decode(best_candidate), best_score


# --------------------------------------------------------------------------- #
# Prompt Injection
# --------------------------------------------------------------------------- #


def prompt_injection(
    model: SimulatedLLM,
    tokenizer: SimulatedTokenizer,
    base_prompt: str,
    injection_text: str,
    target_class: int,
    position: str = "suffix",
    max_length: int = 64,
) -> tuple[Tensor, Tensor]:
    """Prompt injection via context poisoning.

    Perez & Ribeiro, "Ignore This Title and HackAPrompt" (2023).

    Prepends or appends adversarial text to manipulate the model's output.
    The injection text is concatenated with the base prompt and the combined
    sequence is evaluated.

    Args:
        model: a ``SimulatedLLM`` instance in eval mode.
        tokenizer: a ``SimulatedTokenizer`` instance.
        base_prompt: the original user prompt.
        injection_text: adversarial text to inject.
        target_class: desired output class for verification.
        position: where to inject -- ``"prefix"`` or ``"suffix"``.
        max_length: maximum sequence length.

    Returns:
        Tuple of ``(injected_token_ids, logits)`` where injected_token_ids is
        the full tokenized sequence and logits are the model outputs.
    """
    model.eval()

    if position == "prefix":
        combined = injection_text + base_prompt
    else:
        combined = base_prompt + injection_text

    token_ids = tokenizer.encode(combined, max_length=max_length).unsqueeze(0)

    with torch.no_grad():
        logits = model(token_ids)

    return token_ids[0].detach(), logits[0].detach()


# --------------------------------------------------------------------------- #
# Embedding Perturbation
# --------------------------------------------------------------------------- #


def embedding_perturbation(
    model: SimulatedLLM,
    tokenizer: SimulatedTokenizer,
    prompt: str,
    target_class: int,
    epsilon: float = 0.5,
    steps: int = 30,
    alpha: float = 0.05,
    max_length: int = 64,
) -> tuple[Tensor, Tensor]:
    """Continuous perturbation in embedding space.

    Analogous to PGD but operates in the continuous embedding space rather
    than discrete token space. This is a white-box attack that requires access
    to the model's embedding layer.

    The perturbation is bounded in L2 norm by ``epsilon`` in the embedding
    space, following Miyato et al., "Adversarial Training Methods for
    Semi-Supervised Text Classification" (ICLR 2017).

    Args:
        model: a ``SimulatedLLM`` instance.
        tokenizer: a ``SimulatedTokenizer`` instance.
        prompt: input text to perturb.
        target_class: desired model output class.
        epsilon: L2 perturbation budget in embedding space.
        steps: number of PGD steps.
        alpha: per-step size.
        max_length: maximum sequence length.

    Returns:
        Tuple of ``(perturbed_embeddings, logits)`` where perturbed_embeddings
        has shape ``(seq_len, hidden_dim)`` and logits are model outputs.
    """
    model.eval()

    token_ids = tokenizer.encode(prompt, max_length=max_length).unsqueeze(0)
    target = torch.tensor([target_class], dtype=torch.long)

    # Get clean embeddings
    with torch.no_grad():
        clean_emb = model.get_embeddings(token_ids)  # (1, seq, hidden)

    # Initialize perturbation
    delta = torch.zeros_like(clean_emb)

    for _ in range(steps):
        delta = delta.clone().detach().requires_grad_(True)
        perturbed_emb = clean_emb + delta
        logits = model.forward_from_embeddings(perturbed_emb)
        loss = nn.functional.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, delta)[0]

        # PGD step (minimize loss toward target)
        delta = delta.detach() - alpha * grad.sign()

        # Project onto L2 ball
        delta_flat = delta.view(1, -1)
        delta_norm = delta_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factor = (epsilon / delta_norm).clamp(max=1.0)
        delta = (delta_flat * factor).view_as(clean_emb)

    # Final evaluation
    with torch.no_grad():
        perturbed_emb = clean_emb + delta
        logits = model.forward_from_embeddings(perturbed_emb)

    return perturbed_emb[0].detach(), logits[0].detach()


# --------------------------------------------------------------------------- #
# Token Substitution
# --------------------------------------------------------------------------- #


def token_substitution(
    model: SimulatedLLM,
    tokenizer: SimulatedTokenizer,
    prompt: str,
    target_class: int,
    num_substitutions: int = 5,
    max_length: int = 64,
) -> tuple[str, Tensor]:
    """Gradient-guided discrete token-level substitution attack.

    Ebrahimi et al., "HotFlip: White-Box Adversarial Examples for Text
    Classification" (ACL 2018).

    For each position, computes the gradient of the loss with respect to the
    one-hot token input and selects the replacement token that maximally
    reduces loss toward the target class.

    Args:
        model: a ``SimulatedLLM`` instance.
        tokenizer: a ``SimulatedTokenizer`` instance.
        prompt: input text to attack.
        target_class: desired output class.
        num_substitutions: maximum number of tokens to replace.
        max_length: maximum sequence length.

    Returns:
        Tuple of ``(modified_text, logits)`` where modified_text is the
        attacked prompt and logits are the model outputs.
    """
    model.eval()

    token_ids = tokenizer.encode(prompt, max_length=max_length)
    target = torch.tensor([target_class], dtype=torch.long)
    current_tokens = token_ids.clone()

    for _ in range(num_substitutions):
        # Get embeddings and compute gradient
        input_batch = current_tokens.unsqueeze(0)
        emb = model.get_embeddings(input_batch)
        emb_param = emb.clone().detach().requires_grad_(True)
        logits = model.forward_from_embeddings(emb_param)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()

        grad = emb_param.grad[0]  # (seq_len, hidden_dim)

        # Find the position with highest gradient magnitude (most impactful)
        pos_importance = grad.norm(dim=1)  # (seq_len,)
        # Only consider non-padding positions
        non_pad_mask = current_tokens != tokenizer.pad_token_id
        pos_importance = pos_importance * non_pad_mask.float()

        if pos_importance.sum() == 0:
            break

        best_pos = pos_importance.argmax().item()

        # Find best replacement token at that position
        pos_grad = grad[best_pos]  # (hidden_dim,)
        best_token = current_tokens[best_pos].item()
        best_score = float("inf")

        # Evaluate a subset of vocabulary
        candidates = list(range(1, min(tokenizer.vocab_size, 96)))
        for cand_id in candidates:
            cand_tensor = torch.tensor([cand_id], dtype=torch.long)
            cand_emb = model.embedding(cand_tensor)[0]  # (hidden_dim,)
            # Score: alignment with gradient (lower = better for minimizing loss)
            score = (cand_emb * pos_grad).sum().item()
            if score < best_score:
                best_score = score
                best_token = cand_id

        current_tokens[best_pos] = best_token

    # Final evaluation
    with torch.no_grad():
        logits = model(current_tokens.unsqueeze(0))

    modified_text = tokenizer.decode(current_tokens)
    return modified_text, logits[0].detach()


# --------------------------------------------------------------------------- #
# Universal Suffix
# --------------------------------------------------------------------------- #


def universal_suffix(
    models: list[SimulatedLLM],
    tokenizer: SimulatedTokenizer,
    prompts: list[str],
    target_class: int,
    suffix_length: int = 8,
    steps: int = 30,
    max_length: int = 64,
) -> tuple[str, list[float]]:
    """Universal adversarial suffix that transfers across model families.

    Zou et al., "Universal and Transferable Adversarial Attacks on Aligned
    Language Models" (2023).

    Optimizes a single suffix that, when appended to any prompt, causes
    multiple models to predict the target class. The suffix is optimized
    over an ensemble of models and prompts simultaneously.

    Args:
        models: list of ``SimulatedLLM`` instances (representing different
            model families).
        tokenizer: a ``SimulatedTokenizer`` instance.
        prompts: list of prompts to optimize over.
        target_class: desired output class.
        suffix_length: number of suffix tokens.
        steps: number of optimization iterations.
        max_length: maximum sequence length.

    Returns:
        Tuple of ``(suffix_text, per_model_scores)`` where suffix_text is the
        universal suffix and per_model_scores contains the average target-class
        logit for each model.
    """
    for m in models:
        m.eval()

    # Initialize suffix tokens randomly
    suffix_tokens = torch.randint(1, tokenizer.vocab_size, (suffix_length,))

    # Encode all prompts
    encoded_prompts = []
    suffix_starts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, max_length=max_length)
        # Find where to place suffix (after the prompt content)
        prompt_len = min(len(prompt), max_length - suffix_length)
        encoded_prompts.append(tokens)
        suffix_starts.append(prompt_len)

    target = torch.tensor([target_class], dtype=torch.long)

    for step in range(steps):
        # For each suffix position, find the best token across all models/prompts
        for pos_offset in range(suffix_length):
            best_token = suffix_tokens[pos_offset].item()
            best_total_score = float("inf")

            # Evaluate candidates
            num_candidates = min(32, tokenizer.vocab_size - 1)
            candidates = torch.randint(1, tokenizer.vocab_size, (num_candidates,))

            for cand in candidates:
                total_score = 0.0
                trial_suffix = suffix_tokens.clone()
                trial_suffix[pos_offset] = cand.item()

                for model in models:
                    for prompt_tokens, s_start in zip(encoded_prompts, suffix_starts):
                        full_seq = prompt_tokens.clone()
                        end_pos = min(s_start + suffix_length, max_length)
                        actual_len = end_pos - s_start
                        full_seq[s_start:end_pos] = trial_suffix[:actual_len]

                        with torch.no_grad():
                            logits = model(full_seq.unsqueeze(0))
                            # Lower CE loss toward target = better
                            score = nn.functional.cross_entropy(
                                logits, target
                            ).item()
                        total_score += score

                if total_score < best_total_score:
                    best_total_score = total_score
                    best_token = cand.item()

            suffix_tokens[pos_offset] = best_token

    # Final evaluation: compute per-model scores
    per_model_scores = []
    for model in models:
        model_score = 0.0
        count = 0
        for prompt_tokens, s_start in zip(encoded_prompts, suffix_starts):
            full_seq = prompt_tokens.clone()
            end_pos = min(s_start + suffix_length, max_length)
            actual_len = end_pos - s_start
            full_seq[s_start:end_pos] = suffix_tokens[:actual_len]

            with torch.no_grad():
                logits = model(full_seq.unsqueeze(0))
                model_score += logits[0, target_class].item()
                count += 1
        per_model_scores.append(model_score / max(count, 1))

    suffix_text = tokenizer.decode(suffix_tokens)
    return suffix_text, per_model_scores
