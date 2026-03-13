"""Inference logic for EC-CoT and baseline methods."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from src.model import GeminiModel
from src.preprocess import load_truthfulqa, format_question


# [VALIDATOR FIX - Attempt 1]
# [PROBLEM]: 100% accuracy - all predictions return the first valid letter (always "A")
# [CAUSE]: "if letter in text" matches ANY occurrence (e.g., "A" in "ANSWER"), not just the predicted letter
# [FIX]: Use regex to find standalone letter patterns, check "Final Answer:" line, avoid substring matches
#
# [OLD CODE]:
# def extract_answer_letter(text: str, valid_letters: List[str]) -> str:
#     """Extract answer letter from model output."""
#     text = text.strip().upper()
#     if text in valid_letters:
#         return text
#     for letter in valid_letters:
#         if text.startswith(letter):
#             return letter
#     for letter in valid_letters:
#         if letter in text:
#             return letter
#     return valid_letters[0] if valid_letters else "A"
#
# [NEW CODE]:
# [VALIDATOR FIX - Attempt 2]
# [PROBLEM]: All predictions are "A" (100% of samples return the first valid letter)
# [CAUSE]: Gemini API responses don't match expected patterns, always hitting fallback return
# [FIX]: Add logging to diagnose what the model actually returns, handle empty/blocked responses
#
# [OLD CODE]:
# def extract_answer_letter(text: str, valid_letters: List[str]) -> str:
#     """Extract answer letter from model output."""
#     text = text.strip()
#     text_upper = text.upper()
#     ... (pattern matching code)
#     return valid_letters[0] if valid_letters else "A"  # Always returns "A"
#
# [NEW CODE]:
# [VALIDATOR FIX - Attempt 3]
# [PROBLEM]: All predictions are "A" (100% default fallback) - model responses not being parsed
# [CAUSE]: Gemini's actual response format doesn't match any extraction patterns, always hitting fallback
# [FIX]: Add more robust extraction that looks for ANY valid letter in the response, prioritize last occurrence
#
# [OLD CODE]:
# (extensive pattern matching that still missed Gemini's actual response format)
#
# [NEW CODE]:
def extract_answer_letter(
    text: str, valid_letters: List[str], debug: bool = False
) -> str:
    """Extract answer letter from model output with robust fallback strategies."""
    # [VALIDATOR FIX - Attempt 5]
    # [PROBLEM]: All predictions are "A" - pattern matching extracts letter from echoed question choices
    # [CAUSE]: Gemini echoes the question or lists choices (e.g., "A) First option"), and patterns like
    #          r"^([A-E])[\s\.,\)\:]" match "A)" from the choices list, not the actual answer.
    #          Previous attempts looked for ANY letter match, which always found "A" in the first choice.
    # [FIX]: 1) Prioritize answer indicator patterns (Final Answer:, I choose, etc.) over generic letter matches
    #        2) For generic patterns, search from the END of the response backwards (the answer comes last)
    #        3) Ignore choice-list patterns like "A)" that appear early in the response
    #        4) Add pattern for sentences containing "correct" or "is [letter]" near the end
    #
    # [OLD CODE]:
    # (patterns searched in order, first match wins, so "A)" from echoed choices always matched first)
    #
    # [NEW CODE]:
    original_text = text  # Keep for debugging
    text = text.strip()

    # Check for empty or blocked responses - ALWAYS log this
    if not text or len(text) < 1:
        print(f"[EXTRACTION ERROR] Empty response, defaulting to {valid_letters[0]}")
        print(
            f"  This suggests API failures or blocked content. Check model logs above."
        )
        return valid_letters[0] if valid_letters else "A"

    text_upper = text.upper()

    # Try direct match (entire response is just a letter)
    if text_upper in valid_letters:
        if debug:
            print(f"[DEBUG] Direct match: {text_upper}")
        return text_upper

    # PRIORITY 1: Look for explicit answer indicators (high confidence patterns)
    # These should be searched FIRST and are most likely to be the actual answer
    high_confidence_patterns = [
        r"FINAL\s*ANSWER\s*:?\s*\*?\*?([A-E])\*?\*?",  # Final Answer: A or **A**
        r"(?:MY|THE)?\s*ANSWER\s*IS\s*:?\s*\*?\*?([A-E])\*?\*?",  # The answer is A
        r"(?:I\s+)?(?:CHOOSE|SELECT|PICK)\s*:?\s*\*?\*?([A-E])\*?\*?",  # I choose A
        r"CORRECT\s+(?:ANSWER|CHOICE|OPTION)\s+IS\s*:?\s*\*?\*?([A-E])\*?\*?",
        r"\*\*([A-E])\*\*\s*(?:IS|CORRECT)",  # **B** is correct
    ]
    for pattern in high_confidence_patterns:
        match = re.search(pattern, text_upper)
        if match:
            letter = match.group(1)
            if letter in valid_letters:
                if debug:
                    print(f"[DEBUG] High-confidence pattern '{pattern}': {letter}")
                return letter

    # PRIORITY 2: Look for answer at the END of the response (last line/sentence)
    # The actual answer is usually at the end, after the reasoning
    last_200_chars = text_upper[-200:]  # Focus on the last part of the response
    for pattern in high_confidence_patterns:
        match = re.search(pattern, last_200_chars)
        if match:
            letter = match.group(1)
            if letter in valid_letters:
                if debug:
                    print(f"[DEBUG] End-of-response pattern '{pattern}': {letter}")
                return letter

    # PRIORITY 3: Look for standalone letter at end of response (last occurrence)
    # Search backwards from the end to avoid matching echoed question choices
    lines = text_upper.split("\n")
    for line in reversed(lines):  # Check from bottom up
        line = line.strip()
        # Look for a single letter on its own or with minimal surrounding text
        # Avoid lines that look like choice listings (e.g., "A) Some option")
        if line and len(line) <= 10:  # Short lines more likely to be just the answer
            if line[0] in valid_letters and not any(
                marker in line for marker in [")", "(", ".", ":", ";"]
            ):
                if debug:
                    print(f"[DEBUG] Short line at end: {line[0]}")
                return line[0]

        # Look for patterns like "Therefore, C" or "Thus B" near the end
        match = re.search(r"(?:THEREFORE|THUS|SO|HENCE),?\s*([A-E])\s*(?:\.|$)", line)
        if match:
            letter = match.group(1)
            if letter in valid_letters:
                if debug:
                    print(f"[DEBUG] Conclusion pattern: {letter}")
                return letter

    # PRIORITY 4: Find the LAST standalone occurrence of any valid letter
    # This avoids matching "A" from "A) First choice" which appears early
    last_standalone = {}
    for letter in valid_letters:
        # Look for standalone letter (not part of a choice listing like "A)")
        # Match letter with non-letter or boundary before/after
        pattern = r"(?<![A-Z0-9])" + re.escape(letter) + r"(?![A-Z0-9])"
        matches = list(re.finditer(pattern, text_upper))
        if matches:
            # Get position of last match (closest to the end)
            last_standalone[letter] = matches[-1].start()

    if last_standalone:
        # Get the letter that appears last as standalone in the text
        last_letter = max(last_standalone.items(), key=lambda x: x[1])[0]
        if debug:
            print(
                f"[DEBUG] Last standalone letter in text: {last_letter} at position {last_standalone[last_letter]}"
            )
        return last_letter

    # CRITICAL: Absolute fallback should ALWAYS be logged loudly
    print(f"\n{'=' * 60}")
    print(f"[EXTRACTION FALLBACK TRIGGERED - THIS INDICATES A PROBLEM]")
    print(f"Response length: {len(original_text)} chars")
    print(f"Response preview (first 300 chars):\n{original_text[:300]}")
    print(f"Response preview (last 300 chars):\n{original_text[-300:]}")
    print(f"Valid letters: {valid_letters}")
    print(f"Defaulting to: {valid_letters[0]}")
    print(f"{'=' * 60}\n")
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
    example_idx: int = 0,
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

    for i in range(num_self_consistency):
        response = model.generate(prompt, max_tokens=max_tokens)
        cot_responses.append(response)

        # Extract answer (enable debug for first sample of first 3 examples)
        # [VALIDATOR FIX - Attempt 3] Enable debug for ALL samples of first example to diagnose extraction
        debug_mode = (example_idx == 0 and i < 2) or (example_idx < 3 and i == 0)
        if debug_mode:
            print(f"\n[MODEL RESPONSE SAMPLE - Example {example_idx}, Sample {i}]:")
            print(f"Response length: {len(response)} chars")
            print(f"Response: {response[:500] if len(response) > 500 else response}")
            print(f"Valid letters: {valid_letters}")
        answer = extract_answer_letter(response, valid_letters, debug=debug_mode)
        if debug_mode:
            print(f"Extracted: {answer}\n")
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

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: 100% accuracy on TruthfulQA (200/200 correct) - meaningless results
    # [CAUSE]: Need to verify that predictions and ground truth vary across examples
    # [FIX]: Add diagnostic logging to first 5 examples to verify data integrity
    #
    # [OLD CODE]:
    # print(f"Running {method_type} inference on {len(examples)} examples...")
    #
    # [NEW CODE]:
    print(f"Running {method_type} inference on {len(examples)} examples...")

    # Diagnostic: log first 3 examples to verify data
    print("\n=== DIAGNOSTIC: First 3 examples ===")
    for i in range(min(3, len(examples))):
        ex = examples[i]
        print(f"Example {i}:")
        print(f"  Question: {ex['question'][:80]}...")
        print(f"  Choices: {ex['choices']}")
        print(f"  Correct answer: {ex['correct_answer']} (idx={ex['correct_idx']})")
        print(f"  Choice letters: {ex['choice_letters']}")
    print("=" * 50 + "\n")

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
                example_idx=idx,
            )
        else:
            raise ValueError(f"Unknown method type: {method_type}")

        result["example_idx"] = idx
        result["question"] = example["question"]
        results.append(result)

        # Diagnostic: log first 5 predictions
        if idx < 5:
            print(f"\nExample {idx} result:")
            print(f"  Predicted: {result['final_answer']}")
            print(f"  Correct: {result['correct_answer']}")
            print(f"  Match: {result['is_correct']}")
            print(f"  All answers sampled: {result.get('all_answers', [])}")

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

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Sanity validation passed despite 100% accuracy (meaningless results)
    # [CAUSE]: Validation didn't check that outputs are diverse or that accuracy is reasonable
    # [FIX]: Require outputs_unique=True and accuracy in reasonable range (0.2-0.8) for TruthfulQA
    #
    # [OLD CODE]:
    # if len(results) >= 5 and all("final_answer" in r for r in results):
    #     print("SANITY_VALIDATION: PASS")
    # else:
    #     print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples")
    #
    # [NEW CODE]:
    # [VALIDATOR FIX - Attempt 2]
    # [PROBLEM]: Mode was "sanity_check" but code checked for "sanity", so validation didn't execute
    # [CAUSE]: GitHub Actions passes mode="sanity_check" but code expected "sanity"
    # [FIX]: Normalize mode check to handle "sanity_check" or "sanity"
    #
    # [OLD CODE]:
    # if cfg.mode == "sanity":
    #
    # [NEW CODE]:
    # Validation output for sanity/pilot modes
    if cfg.mode in ("sanity", "sanity_check"):
        outputs_unique = len(set(r["final_answer"] for r in results)) > 1
        correct_answers_unique = len(set(r["correct_answer"] for r in results)) > 1

        # [VALIDATOR FIX - Attempt 2]
        # [PROBLEM]: 100% accuracy with all answers="A" - data sampling bias not detected
        # [CAUSE]: random.sample with seed=42 selected a biased subset where nearly all correct answers are "A"
        # [FIX]: Add check for single-letter dominance in both predictions and ground truth
        #
        # [OLD CODE]:
        # validation_summary = { ... }
        #
        # [NEW CODE]:
        # Count letter distributions
        pred_counter = Counter(r["final_answer"] for r in results)
        truth_counter = Counter(r["correct_answer"] for r in results)
        most_common_pred = pred_counter.most_common(1)[0] if pred_counter else ("?", 0)
        most_common_truth = (
            truth_counter.most_common(1)[0] if truth_counter else ("?", 0)
        )

        validation_summary = {
            "samples": len(results),
            "outputs_valid": all("final_answer" in r for r in results),
            "outputs_unique": outputs_unique,
            "correct_answers_unique": correct_answers_unique,
            "accuracy": accuracy,
            "unique_predictions": len(set(r["final_answer"] for r in results)),
            "unique_ground_truth": len(set(r["correct_answer"] for r in results)),
            "most_common_prediction": f"{most_common_pred[0]}({most_common_pred[1]}/{len(results)})",
            "most_common_truth": f"{most_common_truth[0]}({most_common_truth[1]}/{len(results)})",
        }
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(validation_summary)}")

        # Check for suspicious patterns
        fail_reasons = []
        if len(results) < 5:
            fail_reasons.append("insufficient_samples")
        if not all("final_answer" in r for r in results):
            fail_reasons.append("invalid_outputs")
        if not outputs_unique:
            fail_reasons.append("all_predictions_identical")
        if not correct_answers_unique:
            fail_reasons.append("all_ground_truth_identical")
        # [VALIDATOR FIX - Attempt 2] Check for letter dominance (>80% of any single letter)
        if most_common_pred[1] > len(results) * 0.8:
            fail_reasons.append(f"prediction_dominated_by_{most_common_pred[0]}")
        if most_common_truth[1] > len(results) * 0.8:
            fail_reasons.append(f"ground_truth_dominated_by_{most_common_truth[0]}")
        if accuracy > 0.95:
            fail_reasons.append("suspiciously_high_accuracy")
        if accuracy == 0.0:
            fail_reasons.append("zero_accuracy")

        if not fail_reasons:
            print("SANITY_VALIDATION: PASS")
        else:
            print(f"SANITY_VALIDATION: FAIL reason={','.join(fail_reasons)}")

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
