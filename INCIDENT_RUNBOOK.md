# Incident Runbook — Adversarial ML Lab

This runbook covers common incident scenarios for the adversarial ML research lab. Follow the relevant section when an issue arises.

---

## Table of Contents

1. [New Attack Bypasses All Defenses](#1-new-attack-bypasses-all-defenses)
2. [Robustness Regression Detected in CI](#2-robustness-regression-detected-in-ci)
3. [Model Checkpoint Corruption](#3-model-checkpoint-corruption)
4. [CUDA OOM During Adversarial Training](#4-cuda-oom-during-adversarial-training)

---

## 1. New Attack Bypasses All Defenses

### Severity: HIGH

### Symptoms
- A newly added attack achieves near-100% success rate against all existing defenses
- Robust accuracy drops to near-zero under the new threat model
- AutoAttack or custom evaluation shows no defense provides meaningful protection

### Immediate Actions

1. **Confirm the attack is valid** (not a bug):
   ```bash
   # Run the attack in isolation with verbose logging
   python -m attacks.evaluate --attack <new_attack> --model <model_path> --verbose --seed 42

   # Verify perturbation bounds are respected
   python -m attacks.verify_constraints --attack <new_attack> --epsilon 8/255 --norm Linf
   ```

2. **Check for implementation errors**:
   - Verify gradient computation is correct (no detach/no_grad mistakes)
   - Verify perturbation is actually applied to the input (not a copy)
   - Verify the attack is using the correct loss function (CE, not MSE for classification)
   - Run on a known-robust model (e.g., from RobustBench) to validate

3. **Characterize the attack**:
   ```bash
   # Run across multiple models and defenses
   python benchmarks/robustness_regression.py --attack <new_attack> --output attack_report.json

   # Check transferability
   python -m evaluation.transferability --attack <new_attack> --models resnet18,wrn28-10,densenet121
   ```

### Mitigation Steps

4. **Pin the current release** — do NOT include the attack in a release until defenses are evaluated:
   ```bash
   git tag -a v<current>-pre-bypass -m "Last known good before attack bypass"
   ```

5. **Develop counter-defenses**:
   - Try adversarial training with the new attack in the inner loop
   - Test certified defenses (randomized smoothing, interval bound propagation)
   - Evaluate input preprocessing defenses (JPEG compression, spatial smoothing, feature squeezing)
   - Consider ensemble defenses

6. **Update the threat model documentation**:
   - Document the new attack's capabilities and assumptions
   - Update `README.md` with known limitations
   - Add the attack to the evaluation suite

### Resolution Criteria
- [ ] Attack implementation verified as correct
- [ ] At least one defense achieves >20% robust accuracy against the attack
- [ ] Threat model documentation updated
- [ ] Regression benchmark updated with new attack
- [ ] PR reviewed and merged

---

## 2. Robustness Regression Detected in CI

### Severity: MEDIUM

### Symptoms
- CI pipeline fails at the robustness regression gate (`benchmarks/robustness_regression.py`)
- Clean accuracy dropped below 70% threshold
- Robust accuracy dropped below 25% threshold
- Attack generation time exceeds 100ms for 100 samples

### Immediate Actions

1. **Identify the failing assertion**:
   ```bash
   # Run benchmark locally with detailed output
   python benchmarks/robustness_regression.py --output regression_debug.json --device cpu

   # Review the JSON output
   cat regression_debug.json | python -m json.tool
   ```

2. **Check what changed**:
   ```bash
   # Find the commit that caused the regression
   git log --oneline -10
   git diff HEAD~1 -- "*.py"

   # If model weights changed
   git log --oneline -- "checkpoints/" "models/"
   ```

3. **Compare against baseline**:
   ```bash
   # Checkout the last passing commit
   git stash
   git checkout <last-passing-sha>
   python benchmarks/robustness_regression.py --output baseline.json
   git checkout -

   # Compare results
   python -c "
   import json
   baseline = json.load(open('baseline.json'))
   current = json.load(open('regression_debug.json'))
   print(f'Clean acc: {baseline[\"clean_accuracy\"]:.4f} -> {current[\"clean_accuracy\"]:.4f}')
   print(f'Robust acc: {baseline[\"robust_accuracy\"]:.4f} -> {current[\"robust_accuracy\"]:.4f}')
   "
   ```

### Root Cause Analysis

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Clean accuracy dropped | Model architecture change, training bug, data pipeline change | Revert model change, check data loader |
| Robust accuracy dropped | Defense removed/weakened, normalization changed | Verify defense is still applied, check preprocessing |
| Attack time increased | Added complexity to model forward pass, larger model | Profile with `torch.profiler`, optimize or adjust threshold |
| Both accuracies dropped | Weight initialization changed, random seed issue | Set deterministic seeds, verify checkpoint loading |

### Resolution Steps

4. **Fix or revert**:
   ```bash
   # If recent commit is the cause, revert it
   git revert <commit-sha>

   # Or fix forward
   # ... make changes ...
   python benchmarks/robustness_regression.py  # Verify fix locally
   ```

5. **If threshold needs updating** (intentional architecture change):
   - Document why the threshold changed in the PR description
   - Update thresholds in `benchmarks/robustness_regression.py`
   - Get team approval before merging

### Resolution Criteria
- [ ] Benchmark passes locally
- [ ] Root cause identified and documented
- [ ] CI pipeline green
- [ ] No unintentional accuracy loss

---

## 3. Model Checkpoint Corruption

### Severity: MEDIUM-HIGH

### Symptoms
- `torch.load()` raises `RuntimeError`, `UnpicklingError`, or `EOFError`
- Model produces NaN/Inf outputs after loading
- Checkpoint file size is unexpectedly small (truncated write)
- SHA256 hash of checkpoint doesn't match expected value

### Immediate Actions

1. **Verify corruption**:
   ```bash
   python -c "
   import torch
   try:
       ckpt = torch.load('path/to/checkpoint.pt', map_location='cpu', weights_only=True)
       print('Keys:', list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
       print('Load successful')
   except Exception as e:
       print(f'CORRUPTED: {type(e).__name__}: {e}')
   "
   ```

2. **Check for NaN/Inf in weights**:
   ```python
   import torch

   ckpt = torch.load('checkpoint.pt', map_location='cpu', weights_only=True)
   state_dict = ckpt.get('model_state_dict', ckpt)

   for name, param in state_dict.items():
       if torch.isnan(param).any():
           print(f"NaN found in {name}")
       if torch.isinf(param).any():
           print(f"Inf found in {name}")
   ```

3. **Check file integrity**:
   ```bash
   # Check file size
   ls -la path/to/checkpoint.pt

   # Compute hash and compare to known good
   sha256sum path/to/checkpoint.pt

   # Check if file was truncated (compare to expected size)
   file path/to/checkpoint.pt
   ```

### Recovery Steps

4. **Restore from backup**:
   ```bash
   # Check if git-lfs has the file
   git lfs ls-files | grep checkpoint

   # Restore from a previous commit
   git checkout <good-commit> -- checkpoints/model.pt

   # Or restore from cloud backup
   # aws s3 cp s3://bucket/checkpoints/model_backup.pt ./checkpoints/model.pt
   ```

5. **If no backup exists, retrain**:
   ```bash
   # Use the training script with the same config
   python train.py --config configs/model_config.yaml --output checkpoints/model_retrained.pt

   # Verify the retrained model
   python benchmarks/robustness_regression.py --model-path checkpoints/model_retrained.pt
   ```

6. **If corruption was caused by training instability**:
   - Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
   - Add NaN detection in training loop
   - Save checkpoints more frequently
   - Use `torch.save` with `_use_new_zipfile_serialization=True`

### Prevention

- Enable periodic checkpoint validation in training:
  ```python
  # Add to training loop every N epochs:
  if epoch % 5 == 0:
      torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")
      # Verify it loads correctly
      test_load = torch.load(f"checkpoint_epoch_{epoch}.pt", weights_only=True)
      assert all(not torch.isnan(v).any() for v in test_load.values())
  ```
- Store SHA256 hashes alongside checkpoints
- Use git-lfs or DVC for versioned checkpoint management
- Keep at least 3 recent checkpoints

### Resolution Criteria
- [ ] Valid checkpoint restored or retrained
- [ ] Model produces expected accuracy on validation set
- [ ] Root cause of corruption identified
- [ ] Prevention measures implemented

---

## 4. CUDA OOM During Adversarial Training

### Severity: MEDIUM

### Symptoms
- `RuntimeError: CUDA out of memory. Tried to allocate X MiB`
- Training crashes during adversarial example generation (inner loop)
- GPU memory usage spikes during backward pass of adversarial training
- Process killed by OS OOM killer

### Immediate Actions

1. **Get memory snapshot**:
   ```python
   import torch
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
   print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
   torch.cuda.memory_summary()
   ```

2. **Identify the memory spike**:
   ```python
   # Add memory tracking to training loop
   import torch

   torch.cuda.reset_peak_memory_stats()

   # Before adversarial generation
   mem_before = torch.cuda.memory_allocated()

   # ... adversarial training step ...

   # After
   mem_after = torch.cuda.memory_allocated()
   mem_peak = torch.cuda.max_memory_allocated()
   print(f"Before: {mem_before/1e9:.2f}GB, After: {mem_after/1e9:.2f}GB, Peak: {mem_peak/1e9:.2f}GB")
   ```

### Quick Fixes (in order of preference)

3. **Reduce batch size**:
   ```bash
   # Halve the batch size
   python train_adversarial.py --batch-size 64  # was 128
   ```

4. **Use gradient accumulation**:
   ```python
   accumulation_steps = 4
   optimizer.zero_grad()
   for i, (images, labels) in enumerate(loader):
       loss = adversarial_training_step(model, images, labels) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

5. **Use mixed precision (AMP)**:
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   for images, labels in loader:
       optimizer.zero_grad()
       with autocast():
           # Generate adversarial examples in fp16
           adv_images = pgd_attack(model, images, labels, epsilon)
           loss = criterion(model(adv_images), labels)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

6. **Reduce PGD steps or use FGSM-AT**:
   ```python
   # Switch from PGD-7 to FGSM-AT (single step, much less memory)
   # PGD stores computation graph for each step
   adv_images = fgsm_attack(model, images, labels, epsilon)  # 1 step instead of 7
   ```

7. **Free memory aggressively**:
   ```python
   # In the adversarial training loop:
   adv_images = generate_adversarial(model, images, labels, epsilon)
   adv_images = adv_images.detach()  # CRITICAL: detach from computation graph

   # Clear cache periodically
   if batch_idx % 10 == 0:
       torch.cuda.empty_cache()
   ```

8. **Use gradient checkpointing**:
   ```python
   from torch.utils.checkpoint import checkpoint_sequential

   # For deep models, checkpoint intermediate layers
   model.features = checkpoint_sequential(model.features, segments=4)
   ```

### Advanced Solutions

9. **Multi-GPU with DDP**:
   ```bash
   torchrun --nproc_per_node=2 train_adversarial.py --batch-size 128
   ```

10. **Reduce model size for development**:
    ```bash
    # Use a smaller model during development
    python train_adversarial.py --model resnet18  # instead of wrn-28-10
    ```

### Memory Budget Reference

| Model | Clean Training | FGSM-AT | PGD-7-AT | PGD-10-AT |
|-------|---------------|---------|----------|-----------|
| ResNet-18 | 2.5 GB | 4.0 GB | 8.5 GB | 11 GB |
| WRN-28-10 | 5.0 GB | 8.5 GB | 18 GB | 24 GB |
| WRN-70-16 | 12 GB | 20 GB | 40 GB+ | 50 GB+ |

*Batch size 128, CIFAR-10 (32×32), single GPU*

### Resolution Criteria
- [ ] Training completes without OOM
- [ ] Final model achieves expected robust accuracy
- [ ] Memory usage stays within GPU limits with headroom
- [ ] Fix documented for reproducibility

---

## General Incident Response Process

1. **Detect** — CI failure, monitoring alert, or manual observation
2. **Triage** — Assess severity and impact
3. **Contain** — Stop the bleeding (revert, pin, disable)
4. **Investigate** — Root cause analysis
5. **Fix** — Implement and verify the fix
6. **Document** — Update this runbook if needed
7. **Prevent** — Add tests/gates to catch recurrence

## Contact

- Repository maintainers: Check `CODEOWNERS`
- Security issues: Open a private security advisory on GitHub
- CI/Infrastructure: Check `.github/workflows/` for pipeline owners
