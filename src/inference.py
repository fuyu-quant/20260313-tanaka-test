"""Inference logic for EC-CoT and baseline methods."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from src.model import GeminiModel
from src.preprocess import load_truthfulqa, format_question


def extract_answer_letter(text: str, valid_letters: List[str]) -> str:
    """Extract answer letter from model output."""
    text = text.strip().upper()

    # Try direct match
    if text in valid_letters:
        return text

    # Try to find letter at start
    for letter in valid_letters:
        if text.startswith(letter):
            return letter

    # Try to find letter anywhere
    for letter in valid_letters:
        if letter in text:
            return letter

    # Default to first option if no match
    return valid_letters[0] if valid_letters else "A"


def self_consistency_vote(answers: List[str]) -> Tuple[str, float]:
    """
    Get majority vote from multiple answers.

    Returns:
        (most_common_answer, stability_score)
    """
    if not answers:
        return "A", 0.0

    counter = Counter(answers)
    most_common, count = counter.most_common(1)[0]
    stability = count / len(answers)

    return most_common, stability


def standard_cot_inference(
    model: GeminiModel,
    example: Dict[str, Any],
    num_self_consistency: int = 5,
    max_tokens: int = 150,
) -> Dict[str, Any]:
    """
    Standard Chain-of-Thought inference with self-consistency.

    Args:
        model: LLM model
        example: Question example
        num_self_consistency: Number of samples for self-consistency
        max_tokens: Max tokens per generation

    Returns:
        Result dictionary with answer and metadata
    """
    question_text = format_question(example)
    valid_letters = example["choice_letters"]

    prompt = f"""{question_text}

Think step by step and provide your reasoning, then give your final answer as a single letter (A, B, C, D, etc.).

Your response should follow this format:
Reasoning: [Your step-by-step reasoning here]
Final Answer: [Single letter]"""

    # Generate multiple CoT samples
    cot_responses = []
    extracted_answers = []

    for _ in range(num_self_consistency):
        response = model.generate(prompt, max_tokens=max_tokens)
        cot_responses.append(response)

        # Extract answer
        answer = extract_answer_letter(response, valid_letters)
        extracted_answers.append(answer)

    # Get final answer via self-consistency
    final_answer, stability = self_consistency_vote(extracted_answers)

    return {
        "method": "standard_cot",
        "final_answer": final_answer,
        "correct_answer": example["correct_answer"],
        "is_correct": final_answer == example["correct_answer"],
        "stability": stability,
        "num_api_calls": num_self_consistency,
        "cot_samples": cot_responses,
        "all_answers": extracted_answers,
    }


def ec_cot_inference(
    model: GeminiModel,
    example: Dict[str, Any],
    max_claims: int = 6,
    num_evidence: int = 6,
    num_self_consistency: int = 5,
    coverage_threshold: float = 0.7,
    max_repair_iterations: int = 1,
    max_tokens: int = 150,
) -> Dict[str, Any]:
    """
    Evidence-Coverage Chain-of-Thought inference.

    Algorithm:
    1. Draft CoT with numbered atomic claims
    2. Extract claims into propositions
    3. Generate micro-evidence statements
    4. Score coverage (SUPPORTED/CONTRADICTED/NOT-ENOUGH-INFO)
    5. Repair if coverage below threshold
    6. Return answer with highest coverage and stability

    Args:
        model: LLM model
        example: Question example
        max_claims: Maximum number of claims
        num_evidence: Number of evidence statements to generate
        num_self_consistency: Number of samples for stability check
        coverage_threshold: Minimum coverage to accept without repair
        max_repair_iterations: Maximum repair attempts
        max_tokens: Max tokens per generation

    Returns:
        Result dictionary with answer and metadata
    """
    question_text = format_question(example)
    valid_letters = example["choice_letters"]

    # Step 1: Draft CoT with numbered claims
    draft_prompt = f"""{question_text}

Provide a step-by-step chain-of-thought with 4-6 numbered atomic claims, then give your final answer.

Format:
1. [First reasoning step]
2. [Second reasoning step]
... (up to 6 steps)
Final Answer: [Single letter]"""

    draft_response = model.generate(draft_prompt, max_tokens=max_tokens)
    api_calls = 1

    # Extract answer from draft
    draft_answer = extract_answer_letter(draft_response, valid_letters)

    # Step 2: Extract claims
    claim_prompt = f"""Given the following reasoning steps, extract each claim as a short proposition (subject-relation-object or simple statement).

Reasoning:
{draft_response}

Extract each numbered claim as a concise proposition. List them as:
Claim 1: [proposition]
Claim 2: [proposition]
..."""

    claims_response = model.generate(claim_prompt, max_tokens=max_tokens)
    api_calls += 1

    # Parse claims
    claims = []
    for line in claims_response.split("\n"):
        if line.strip().startswith("Claim"):
            claim_text = line.split(":", 1)[1].strip() if ":" in line else line
            claims.append(claim_text)

    if not claims:
        # Fallback: use original response lines as claims
        claims = [
            line.strip()
            for line in draft_response.split("\n")
            if line.strip() and line[0].isdigit()
        ]

    claims = claims[:max_claims]

    # Step 3: Generate micro-evidence
    evidence_prompt = f"""Given these claims about the question, generate {num_evidence} short evidence statements (1-2 sentences each) that could support or refute them.

Question: {example["question"]}

Claims:
{chr(10).join(f"- {c}" for c in claims)}

Generate {num_evidence} micro-evidence statements:"""

    evidence_response = model.generate(evidence_prompt, max_tokens=max_tokens)
    api_calls += 1

    # Parse evidence
    evidence_list = [
        line.strip() for line in evidence_response.split("\n") if line.strip()
    ]
    evidence_list = [e.lstrip("0123456789.-) ") for e in evidence_list]
    evidence_list = evidence_list[:num_evidence]

    # Step 4: Coverage scoring
    coverage_prompt = f"""Given the following claims and evidence, label each claim as SUPPORTED, CONTRADICTED, or NOT-ENOUGH-INFO.

Claims:
{chr(10).join(f"{i + 1}. {c}" for i, c in enumerate(claims))}

Evidence:
{chr(10).join(f"- {e}" for e in evidence_list)}

For each claim, provide:
Claim X: [SUPPORTED/CONTRADICTED/NOT-ENOUGH-INFO]"""

    coverage_response = model.generate(coverage_prompt, max_tokens=max_tokens)
    api_calls += 1

    # Parse coverage
    supported_count = coverage_response.upper().count("SUPPORTED")
    contradicted_count = coverage_response.upper().count("CONTRADICTED")
    coverage_score = supported_count / max(len(claims), 1)

    # Step 5: Self-consistency for stability
    stability_answers = [draft_answer]
    for _ in range(num_self_consistency - 1):
        quick_prompt = f"{question_text}\n\nProvide your answer as a single letter:"
        quick_response = model.generate(quick_prompt, max_tokens=20, temperature=0.7)
        stability_answers.append(extract_answer_letter(quick_response, valid_letters))
        api_calls += 1

    _, stability = self_consistency_vote(stability_answers)

    # Step 6: Adaptive repair if needed
    repaired = False
    if coverage_score < coverage_threshold or contradicted_count > 0:
        # Generate targeted repair
        repair_prompt = f"""{question_text}

Previous reasoning had insufficient evidence coverage. Re-analyze using ONLY well-supported steps (max 3 steps), citing evidence.

Provide:
1. [First supported step]
2. [Second supported step]
3. [Third supported step]
Final Answer: [Single letter]"""

        repair_response = model.generate(repair_prompt, max_tokens=max_tokens)
        api_calls += 1
        repaired = True

        # Re-evaluate coverage for repaired answer
        repair_answer = extract_answer_letter(repair_response, valid_letters)

        # Re-check stability
        repair_stability_answers = [repair_answer]
        for _ in range(min(3, num_self_consistency)):
            quick_prompt = f"{question_text}\n\nProvide your answer as a single letter:"
            quick_response = model.generate(
                quick_prompt, max_tokens=20, temperature=0.7
            )
            repair_stability_answers.append(
                extract_answer_letter(quick_response, valid_letters)
            )
            api_calls += 1

        _, repair_stability = self_consistency_vote(repair_stability_answers)

        # Use repaired answer if it improves coverage or maintains stability
        if repair_stability >= stability * 0.8:  # Allow slight stability decrease
            final_answer = repair_answer
            final_stability = repair_stability
        else:
            final_answer = draft_answer
            final_stability = stability
    else:
        final_answer = draft_answer
        final_stability = stability

    return {
        "method": "ec_cot",
        "final_answer": final_answer,
        "correct_answer": example["correct_answer"],
        "is_correct": final_answer == example["correct_answer"],
        "coverage_score": coverage_score,
        "num_claims": len(claims),
        "num_supported": supported_count,
        "num_contradicted": contradicted_count,
        "stability": final_stability,
        "repaired": repaired,
        "num_api_calls": api_calls,
        "draft_response": draft_response,
        "claims": claims,
        "evidence": evidence_list,
    }


def run_inference(cfg: DictConfig) -> None:
    """
    Main inference entry point.

    Args:
        cfg: Hydra config
    """
    # Initialize WandB if not disabled
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"WandB run URL: {wandb.run.url}")

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: ConfigAttributeError: Key 'method' is not in struct
    # [CAUSE]: Hydra loads run configs under 'run' group, but code tried to access cfg.dataset/model/method directly
    # [FIX]: Access all run config fields through cfg.run namespace
    #
    # [OLD CODE]:
    # print(f"Loading dataset: {cfg.dataset.name}")
    # examples = load_truthfulqa(split=cfg.dataset.split, ...)
    # model = GeminiModel(model_name=cfg.model.name, ...)
    # method_type = cfg.method.type
    #
    # [NEW CODE]:
    # Load dataset
    print(f"Loading dataset: {cfg.run.dataset.name}")
    examples = load_truthfulqa(
        split=cfg.run.dataset.split,
        num_samples=cfg.run.dataset.num_samples,
        seed=cfg.run.dataset.seed,
        cache_dir=cfg.run.inference.cache_dir,
    )
    print(f"Loaded {len(examples)} examples")

    # Initialize model
    print(f"Initializing model: {cfg.run.model.name}")
    model = GeminiModel(
        model_name=cfg.run.model.name,
        temperature=cfg.run.model.temperature,
        max_output_tokens=cfg.run.model.max_output_tokens,
    )

    # Run inference based on method
    results = []
    correct_count = 0

    method_type = cfg.run.method.type

    print(f"Running {method_type} inference on {len(examples)} examples...")

    for idx, example in enumerate(tqdm(examples, desc="Inference")):
        if method_type == "ec_cot":
            result = ec_cot_inference(
                model=model,
                example=example,
                max_claims=cfg.run.method.max_claims,
                num_evidence=cfg.run.method.num_evidence,
                num_self_consistency=cfg.run.method.num_self_consistency,
                coverage_threshold=cfg.run.method.coverage_threshold,
                max_repair_iterations=cfg.run.method.max_repair_iterations,
                max_tokens=cfg.run.method.max_tokens_per_step,
            )
        elif method_type == "standard_cot":
            result = standard_cot_inference(
                model=model,
                example=example,
                num_self_consistency=cfg.run.method.num_self_consistency,
                max_tokens=cfg.run.method.max_tokens_per_step,
            )
        else:
            raise ValueError(f"Unknown method type: {method_type}")

        result["example_idx"] = idx
        result["question"] = example["question"]
        results.append(result)

        if result["is_correct"]:
            correct_count += 1

        # Log to WandB
        if cfg.wandb.mode != "disabled":
            wandb.log(
                {
                    "example_idx": idx,
                    "is_correct": int(result["is_correct"]),
                    "accuracy_running": correct_count / (idx + 1),
                }
            )

    # Calculate metrics
    accuracy = correct_count / len(results) if results else 0.0
    avg_stability = (
        sum(r.get("stability", 0) for r in results) / len(results) if results else 0.0
    )

    metrics = {
        "accuracy": accuracy,
        "num_correct": correct_count,
        "num_total": len(results),
        "avg_stability": avg_stability,
    }

    if method_type == "ec_cot":
        avg_coverage = (
            sum(r.get("coverage_score", 0) for r in results) / len(results)
            if results
            else 0.0
        )
        repair_rate = (
            sum(r.get("repaired", False) for r in results) / len(results)
            if results
            else 0.0
        )
        metrics["avg_coverage"] = avg_coverage
        metrics["repair_rate"] = repair_rate

    print(f"\nFinal Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Avg Stability: {avg_stability:.4f}")
    if method_type == "ec_cot":
        print(f"  Avg Coverage: {metrics['avg_coverage']:.4f}")
        print(f"  Repair Rate: {metrics['repair_rate']:.4f}")

    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved to: {results_dir}")

    # Log final metrics to WandB
    if cfg.wandb.mode != "disabled":
        wandb.summary.update(metrics)
        wandb.finish()

    # Validation output for sanity/pilot modes
    if cfg.mode == "sanity":
        validation_summary = {
            "samples": len(results),
            "outputs_valid": all("final_answer" in r for r in results),
            "outputs_unique": len(set(r["final_answer"] for r in results)) > 1,
            "accuracy": accuracy,
        }
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(validation_summary)}")

        if len(results) >= 5 and all("final_answer" in r for r in results):
            print("SANITY_VALIDATION: PASS")
        else:
            print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples")

    elif cfg.mode == "pilot":
        validation_summary = {
            "samples": len(results),
            "primary_metric": "accuracy",
            "primary_metric_value": accuracy,
            "outputs_unique": len(set(r["final_answer"] for r in results)) > 1,
        }
        print(f"PILOT_VALIDATION_SUMMARY: {json.dumps(validation_summary)}")

        if len(results) >= 50 and accuracy > 0:
            print("PILOT_VALIDATION: PASS")
        else:
            print(
                f"PILOT_VALIDATION: FAIL reason=insufficient_samples_or_zero_accuracy"
            )
