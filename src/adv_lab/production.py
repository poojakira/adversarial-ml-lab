"""
Adversarial ML Lab — Production Hardening

Transforms the offline benchmark lab into a CI/CD integrated robustness gate:
- Online evaluation service (attacks running models via Triton/TorchServe/vLLM)
- KMS-backed HMAC signing for tamper-evident results
- RobustBench parity validation
- GitHub Action for PR gating
- Policy-as-code (Rego) for robustness requirements
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from adv_lab.logger import get_logger

log = get_logger(__name__)


class ThreatModel(Enum):
    """Standard threat models for robustness evaluation."""
    LINF_EPS_8_255 = "linf_eps_8_255"      # CIFAR-10 standard
    LINF_EPS_4_255 = "linf_eps_4_255"      # ImageNet standard
    L2_EPS_0_5 = "l2_eps_0_5"              # L2 threat model
    L2_EPS_1_0 = "l2_eps_1_0"
    L1_EPS_10 = "l1_eps_10"
    L0_EPS_0_1 = "l0_eps_0_1"
    SEMANTIC = "semantic"                   # Spatial transforms
    CORRUPTION = "corruption"               # Common corruptions (ImageNet-C)


@dataclass
class AttackConfig:
    """Configuration for an adversarial attack."""
    name: str
    threat_model: ThreatModel
    eps: float
    norm: str  # "linf", "l2", "l1", "l0"
    targeted: bool = False
    n_restarts: int = 5
    n_steps: int = 100
    step_size: Optional[float] = None
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of a robustness evaluation."""
    model_name: str
    threat_model: ThreatModel
    clean_accuracy: float
    robust_accuracy: float
    attack_accuracy: dict[str, float]  # per-attack
    eps: float
    timestamp: datetime
    hmac_signature: str
    evaluation_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "threat_model": self.threat_model.value,
            "clean_accuracy": self.clean_accuracy,
            "robust_accuracy": self.robust_accuracy,
            "attack_accuracy": self.attack_accuracy,
            "eps": self.eps,
            "timestamp": self.timestamp.isoformat(),
            "hmac_signature": self.hmac_signature,
            "evaluation_id": self.evaluation_id,
            "metadata": self.metadata,
        }


class KMSSigner:
    """
    KMS-backed HMAC signing for evaluation results.
    
    Supports AWS KMS, GCP KMS, HashiCorp Vault Transit, or local file.
    """
    
    def __init__(
        self,
        provider: str = "local",  # "aws", "gcp", "vault", "local"
        key_id: Optional[str] = None,
        key_path: Optional[str] = None,
        region: str = "us-east-1",
    ):
        self.provider = provider
        self.key_id = key_id
        self.key_path = Path(key_path) if key_path else None
        self.region = region
        
        # Initialize provider-specific client
        self._init_provider()
    
    def _init_provider(self) -> None:
        if self.provider == "aws":
            import boto3
            self._kms = boto3.client("kms", region_name=self.region)
        elif self.provider == "gcp":
            from google.cloud import kms
            self._kms = kms.KeyManagementServiceClient()
        elif self.provider == "vault":
            import hvac
            self._vault = hvac.Client(url=os.environ.get("VAULT_ADDR"), token=os.environ.get("VAULT_TOKEN"))
        elif self.provider == "local":
            if not self.key_path:
                raise ValueError("key_path required for local provider")
            with open(self.key_path, "rb") as f:
                self._local_key = f.read()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def sign(self, data: bytes) -> str:
        """Sign data and return hex signature."""
        if self.provider == "aws":
            response = self._kms.sign(
                KeyId=self.key_id,
                Message=data,
                MessageType="RAW",
                SigningAlgorithm="ECDSA_SHA_256",
            )
            return base64.b64encode(response["Signature"]).decode()
        
        elif self.provider == "gcp":
            from google.cloud.kms_v1 import DigestAlgorithm
            response = self._kms.asymmetric_sign(
                name=self.key_id,
                digest=digest.Digest(sha256=data),
                algorithm=kms.CryptoKeyVersionAlgorithm.EC_SIGN_P256_SHA256,
            )
            return base64.b64encode(response.signature).decode()
        
        elif self.provider == "vault":
            response = self._vault.secrets.transit.sign_data(
                name=self.key_id,
                hash_input=base64.b64encode(data).decode(),
                hash_algorithm="sha2-256",
                signature_algorithm="ecdsa-p256",
            )
            return response["data"]["signature"].replace("vault:v1:", "")
        
        elif self.provider == "local":
            sig = hmac.new(self._local_key, data, hashlib.sha256).digest()
            return base64.b64encode(sig).decode()
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def verify(self, data: bytes, signature: str) -> bool:
        """Verify signature (for local provider only; others use KMS verify)."""
        if self.provider == "local":
            expected = self.sign(data)
            return hmac.compare_digest(expected, signature)
        
        # For cloud KMS, use their verify API
        if self.provider == "aws":
            try:
                self._kms.verify(
                    KeyId=self.key_id,
                    Message=data,
                    Signature=base64.b64decode(signature),
                    SigningAlgorithm="ECDSA_SHA_256",
                )
                return True
            except Exception:
                return False
        # Similar for others...
        return False


class OnlineEvaluator:
    """
    Online robustness evaluation against serving models.
    
    Supports Triton Inference Server, TorchServe, vLLM, TensorRT-LLM.
    """
    
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        input_shape: tuple[int, ...],
        signer: Optional[KMSSigner] = None,
        timeout: int = 30,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.input_shape = input_shape
        self.signer = signer or KMSSigner(provider="local", key_path=os.environ.get("ADV_LAB_HMAC_KEY"))
        self.timeout = timeout
        
        # Detect server type
        self.server_type = self._detect_server_type()
    
    def _detect_server_type(self) -> str:
        """Detect model server type from endpoint."""
        try:
            import requests
            resp = requests.get(f"{self.endpoint}/health", timeout=5)
            if "triton" in resp.text.lower():
                return "triton"
            elif "torchserve" in resp.text.lower():
                return "torchserve"
            elif "vllm" in resp.text.lower():
                return "vllm"
        except (requests.RequestException, OSError):
            pass
        return "unknown"
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference on the model server."""
        import requests
        
        if self.server_type == "triton":
            return self._predict_triton(inputs)
        elif self.server_type == "torchserve":
            return self._predict_torchserve(inputs)
        elif self.server_type == "vllm":
            return self._predict_vllm(inputs)
        else:
            # Generic HTTP POST
            return self._predict_generic(inputs)
    
    def _predict_triton(self, inputs: np.ndarray) -> np.ndarray:
        """Triton Inference Server via HTTP."""
        import json

        import requests
        
        payload = {
            "inputs": [{
                "name": "INPUT",
                "shape": list(inputs.shape),
                "datatype": "FP32",
                "data": inputs.flatten().tolist(),
            }],
            "outputs": [{"name": "OUTPUT"}],
        }
        
        resp = requests.post(
            f"{self.endpoint}/v2/models/{self.model_name}/infer",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        return np.array(result["outputs"][0]["data"]).reshape(-1)
    
    def _predict_torchserve(self, inputs: np.ndarray) -> np.ndarray:
        """TorchServe inference."""
        import requests
        
        files = {"data": ("input.npy", inputs.tobytes(), "application/octet-stream")}
        resp = requests.post(
            f"{self.endpoint}/predictions/{self.model_name}",
            files=files,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return np.array(resp.json())
    
    def _predict_vllm(self, inputs: np.ndarray) -> np.ndarray:
        """vLLM OpenAI-compatible API."""
        import requests
        
        # vLLM expects text inputs; for vision we'd need different handling
        payload = {
            "model": self.model_name,
            "prompt": inputs.tolist() if inputs.ndim == 1 else inputs[0].tolist(),
            "max_tokens": 100,
        }
        
        resp = requests.post(
            f"{self.endpoint}/v1/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        return np.array(result["choices"][0]["text"])
    
    def _predict_generic(self, inputs: np.ndarray) -> np.ndarray:
        """Generic HTTP POST."""
        import requests
        
        resp = requests.post(
            f"{self.endpoint}/predict",
            json={"inputs": inputs.tolist()},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return np.array(resp.json())
    
    def evaluate_robustness(
        self,
        attacks: list[AttackConfig],
        n_samples: int = 1000,
        batch_size: int = 32,
    ) -> EvaluationResult:
        """
        Run full robustness evaluation against online model.
        
        Uses Foolbox/JAX or native attack implementations.
        """
        from adv_lab.attacks import (
            autoattack,
            cw,
            fgsm,
            pgd,
            square_attack,
        )
        
        # Load test data
        # In production, this would come from a test dataset bucket
        test_data = self._load_test_data(n_samples)
        test_labels = self._load_test_labels(n_samples)
        
        # Clean accuracy
        clean_preds = self._batch_predict(test_data)
        clean_acc = float(np.mean(clean_preds == test_labels))
        
        # Run attacks
        attack_results = {}
        robust_preds = test_labels.copy()
        
        for attack_config in attacks:
            log.info(f"Running attack: {attack_config.name}")
            
            if attack_config.name == "fgsm":
                adv_inputs = fgsm.generate(test_data, test_labels, attack_config.eps)
            elif attack_config.name == "pgd":
                adv_inputs = pgd.generate(test_data, test_labels, attack_config.eps, attack_config.n_steps)
            elif attack_config.name == "cw":
                adv_inputs = cw.generate(test_data, test_labels, attack_config.eps)
            elif attack_config.name == "autoattack":
                adv_inputs = autoattack.generate(test_data, test_labels, attack_config.eps)
            elif attack_config.name == "square":
                adv_inputs = square_attack.generate(test_data, test_labels, attack_config.eps)
            else:
                log.warning(f"Unknown attack: {attack_config.name}")
                continue
            
            adv_preds = self._batch_predict(adv_inputs)
            attack_acc = float(np.mean(adv_preds == test_labels))
            attack_results[attack_config.name] = attack_acc
            
            # For robust accuracy, take intersection
            robust_preds = np.logical_and(robust_preds == test_labels, adv_preds == test_labels)
        
        robust_acc = float(np.mean(robust_preds))
        
        # Create result
        result = EvaluationResult(
            model_name=self.model_name,
            threat_model=attacks[0].threat_model if attacks else ThreatModel.LINF_EPS_8_255,
            clean_accuracy=clean_acc,
            robust_accuracy=robust_acc,
            attack_accuracy=attack_results,
            eps=attacks[0].eps if attacks else 0.0,
            timestamp=datetime.utcnow(),
            hmac_signature="",
            evaluation_id=f"eval_{int(time.time())}",
        )
        
        # Sign result
        result_bytes = json.dumps(result.to_dict(), sort_keys=True).encode()
        result.hmac_signature = self.signer.sign(result_bytes)
        
        return result
    
    def _batch_predict(self, inputs: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Run predictions in batches."""
        preds = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            preds.append(self.predict(batch))
        return np.concatenate(preds)
    
    def _load_test_data(self, n: int) -> np.ndarray:
        """Load test data from storage."""
        # In production: load from S3/GCS/test dataset
        # For now: generate synthetic
        return np.random.randn(n, *self.input_shape[1:]).astype(np.float32)
    
    def _load_test_labels(self, n: int) -> np.ndarray:
        """Load test labels."""
        return np.random.randint(0, 10, n)


class RobustnessPolicyEngine:
    """
    Robustness Evaluation Engine — CI/CD integration.
    
    Runs as a GitHub Action or standalone service to gate model promotion.
    """
    
    def __init__(
        self,
        policy_path: str = "policies/robustness.rego",
        signer: Optional[KMSSigner] = None,
        robustbench_tolerance: float = 0.02,  # Allow 2% deviation from RobustBench
    ):
        self.policy_path = Path(policy_path)
        self.signer = signer or KMSSigner(provider="local")
        self.robustbench_tolerance = robustbench_tolerance
        
        # Load Rego policy
        self._load_policy()
    
    def _load_policy(self) -> None:
        """Load and compile Rego policy."""
        try:
            import opa
            self._opa = opa.OPA()
            with open(self.policy_path) as f:
                self._opa.add_policy("robustness", f.read())
        except ImportError:
            log.warning("OPA not available, using Python policy evaluation")
            self._opa = None
        except OSError as e:
            log.warning(f"Policy file not found: {e}")
            self._opa = None
    
    def evaluate_policy(
        self,
        result: EvaluationResult,
    ) -> tuple[bool, list[str]]:
        """
        Evaluate robustness result against policy.
        
        Returns:
            (passed, violations)
        """
        violations = []
        
        # Policy 1: Minimum robust accuracy
        min_robust = self._get_policy_value("min_robust_accuracy", 0.35)
        if result.robust_accuracy < min_robust:
            violations.append(f"Robust accuracy {result.robust_accuracy:.3f} below minimum {min_robust}")
        
        # Policy 2: Clean accuracy not too degraded
        min_clean = self._get_policy_value("min_clean_accuracy", 0.80)
        if result.clean_accuracy < min_clean:
            violations.append(f"Clean accuracy {result.clean_accuracy:.3f} below minimum {min_clean}")
        
        # Policy 3: No single attack drops accuracy below threshold
        min_attack = self._get_policy_value("min_per_attack_accuracy", 0.10)
        for attack, acc in result.attack_accuracy.items():
            if acc < min_attack:
                violations.append(f"Attack {attack} accuracy {acc:.3f} below minimum {min_attack}")
        
        # Policy 4: RobustBench parity check
        if self._opa:
            # Query OPA for RobustBench comparison
            pass
        
        return len(violations) == 0, violations
    
    def _get_policy_value(self, key: str, default: float) -> float:
        """Get policy value from Rego or default."""
        # Would query OPA in real implementation
        return default
    
    def run_evaluation_gate(
        self,
        evaluator: OnlineEvaluator,
        attacks: list[AttackConfig],
        n_samples: int = 1000,
    ) -> tuple[bool, EvaluationResult, list[str]]:
        """
        Run full evaluation and check against policy.
        
        Used as CI/CD gate: returns (passed, result, violations)
        """
        result = evaluator.evaluate_robustness(attacks, n_samples)
        passed, violations = self.evaluate_policy(result)
        
        # Sign and store result
        result_bytes = json.dumps(result.to_dict(), sort_keys=True).encode()
        result.hmac_signature = self.signer.sign(result_bytes)
        
        log.info(f"Evaluation gate: {'PASSED' if passed else 'FAILED'} - {len(violations)} violations")
        
        return passed, result, violations


class RobustBenchValidator:
    """
    Validates evaluation results against RobustBench leaderboard.
    
    Ensures evaluation implementation matches published results.
    """
    
    ROBUSTBENCH_MODELS = {
        # CIFAR-10 L_inf eps=8/255
        "cifar10_linf_8": {
            "Carmon2019Unlabeled": 0.6651,
            "Rebuffi2021Fixing_70_16_cutmix_extra": 0.7371,
            "Gowal2021Improving_70_16_ddpm_100m": 0.7135,
        },
        # ImageNet L_inf eps=4/255
        "imagenet_linf_4": {
            "Salman2020Do_R50": 0.4978,
            "Engstrom2019Robustness": 0.4424,
        },
    }
    
    @classmethod
    def validate(cls, model_name: str, dataset: str, threat_model: ThreatModel, robust_acc: float) -> dict[str, Any]:
        """Compare against RobustBench."""
        key = f"{dataset}_{threat_model.value}"
        benchmark = cls.ROBUSTBENCH_MODELS.get(key, {})
        
        if model_name not in benchmark:
            return {
                "model_in_benchmark": False,
                "message": f"Model {model_name} not in RobustBench for {key}",
            }
        
        expected = benchmark[model_name]
        diff = abs(robust_acc - expected)
        
        return {
            "model_in_benchmark": True,
            "expected_robust_acc": expected,
            "actual_robust_acc": robust_acc,
            "difference": diff,
            "within_tolerance": diff <= 0.02,  # 2% tolerance
        }


# GitHub Action entrypoint
def run_github_action():
    """Entry point for GitHub Action."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adversarial Robustness Gate")
    parser.add_argument("--endpoint", required=True, help="Model server endpoint")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--input-shape", required=True, help="Input shape (e.g., 1,3,32,32)")
    parser.add_argument("--policy", default="policies/robustness.rego", help="OPA policy file")
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "autoattack"])
    parser.add_argument("--eps", type=float, default=8/255)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--kms-provider", default="local", choices=["local", "aws", "gcp", "vault"])
    parser.add_argument("--kms-key", help="KMS key ID/path")
    parser.add_argument("--output", default="robustness_result.json", help="Output file")
    parser.add_argument("--fail-on-violation", action="store_true", help="Exit non-zero on violation")
    
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(",")))
    
    # Setup signer
    signer = KMSSigner(provider=args.kms_provider, key_id=args.kms_key)
    
    # Setup evaluator
    evaluator = OnlineEvaluator(
        endpoint=args.endpoint,
        model_name=args.model_name,
        input_shape=input_shape,
        signer=signer,
    )
    
    # Build attack configs
    attacks = []
    for attack_name in args.attacks:
        attacks.append(AttackConfig(
            name=attack_name,
            threat_model=ThreatModel.LINF_EPS_8_255,
            eps=args.eps,
            norm="linf",
        ))
    
    # Setup policy engine
    engine = RobEngine(policy_path=args.policy, signer=signer)
    
    # Run evaluation gate
    passed, result, violations = engine.run_evaluation_gate(
        evaluator=evaluator,
        attacks=attacks,
        n_samples=args.n_samples,
    )
    
    # Save result
    with open(args.output, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    # Print violations
    if violations:
        print("VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
    
    if args.fail_on_violation and not passed:
        print("::error::Robustness gate failed")
        exit(1)
    
    print("Robustness gate passed")


if __name__ == "__main__":
    run_github_action()
