package advlab.robustness

# Robustness Policy for Adversarial ML Lab
# ========================================
# Policy-as-code for CI/CD gates on adversarial robustness
# Implements thresholds for different attack types and datasets

import future.keywords.in
import future.keywords.if

# Main allow decision
allow := {
    all_gates_pass
    violations := get_violations()
    count(violations) == 0
}

# Get all violations
violations := get_violations()

# Get violations as strings for reporting
get_violations() := [v |
    v := sprintf(\"Gate failed: %s (value: %.4f, threshold: %.4f)\", [gate.name, gate.value, gate.threshold])
    gate := gate_results[_]
    not gate.passed
]

# Individual gate results
gate_results := [gate |
    gate := pgd_linf_gate
]
gate_results := gate_results + [gate |
    gate := autoattack_gate
]
gate_results := gate_results + [gate |
    gate := clean_accuracy_gate
]
gate_results := gate_results + [gate |
    gate := robust_accuracy_gate
]

# PGD L-inf gate (most common benchmark)
pgd_linf_gate := {
    \"name\": \"PGD L-inf epsilon=8/255\",
    \"threshold\": input.threshold,
    \"value\": input.robust_accuracy,
    \"passed\": input.robust_accuracy >= input.threshold,
    \"attack\": \"pgd\",
    \"norm\": \"linf\",
    \"epsilon\": 8/255,
}

# AutoAttack gate
autoattack_gate := {
    \"name\": \"AutoAttack standard\",
    \"threshold\": input.threshold,
    \"value\": input.robust_accuracy,
    \"passed\": input.robust_accuracy >= input.threshold,
    \"attack\": \"autoattack\",
    \"norm\": \"linf\",
    \"epsilon\": 8/255,
}

# Clean accuracy gate (must maintain reasonable clean accuracy)
clean_accuracy_gate := {
    \"name\": \"Clean accuracy minimum\",
    \"threshold\": 0.80,
    \"value\": input.clean_accuracy,
    \"passed\": input.clean_accuracy >= 0.80,
    \"attack\": \"none\",
}

# Robust accuracy gate (general)
robust_accuracy_gate := {
    \"name\": \"Robust accuracy threshold\",
    \"threshold\": input.threshold,
    \"value\": input.robust_accuracy,
    \"passed\": input.robust_accuracy >= input.threshold,
    \"attack\": \"all\",
}

# Dataset-specific thresholds
dataset_thresholds := {
    \"cifar10\": {
        \"linf\": {
            \"8/255\": 0.57,
            \"16/255\": 0.45,
            \"32/255\": 0.30,
        },
        \"l2\": {
            \"0.5\": 0.75,
            \"1.0\": 0.65,
            \"2.0\": 0.50,
        },
    },
    \"cifar100\": {
        \"linf\": {
            \"8/255\": 0.35,
            \"16/255\": 0.25,
        },
        \"l2\": {
            \"0.5\": 0.55,
        },
    },
    \"imagenet\": {
        \"linf\": {
            \"4/255\": 0.45,
            \"8/255\": 0.35,
        },
        \"l2\": {
            \"0.5\": 0.60,
        },
    },
}

# RobustBench parity check
robustbench_parity := {
    \"target\": dataset_thresholds[input.dataset][input.norm][sprintf(\"%v\", input.epsilon)],
    \"actual\": input.robust_accuracy,
    \"within_tolerance\": abs(dataset_thresholds[input.dataset][input.norm][sprintf(\"%v\", input.epsilon)] - input.robust_accuracy) <= 0.02,
}

# Attack-specific requirements
attack_requirements := {
    \"pgd\": {
        \"min_steps\": 10,
        \"min_restarts\": 5,
        \"norm\": \"linf\",
    },
    \"autoattack\": {
        \"version\": \"standard\",
        \"norm\": \"linf\",
    },
    \"fab\": {
        \"norm\": [\"linf\", \"l2\"],
    },
    \"square\": {
        \"norm\": [\"linf\", \"l2\"],
        \"queries\": 5000,
    },
}

# Check if required attacks are present
required_attacks_present := {
    \"pgd\" in split(input.attacks, \",\")
    \"autoattack\" in split(input.attacks, \",\")
}

# Minimum sample size check
sufficient_samples := input.num_samples >= 100

# Overall compliance
all_gates_pass := {
    pgd_linf_gate.passed
    autoattack_gate.passed
    clean_accuracy_gate.passed
    robust_accuracy_gate.passed
    required_attacks_present
    sufficient_samples
}

# Helper functions
split(s, delim) := result {
    result := split_string(s, delim)
}

abs(x) := x if x >= 0 else -x

# Default input values for testing
default input := {
    \"model\": \"test-model\",
    \"dataset\": \"cifar10\",
    \"attacks\": \"pgd,autoattack\",
    \"epsilon\": 8/255,
    \"norm\": \"linf\",
    \"robust_accuracy\": 0.6,
    \"clean_accuracy\": 0.9,
    \"threshold\": 0.35,
    \"num_samples\": 1000,
}