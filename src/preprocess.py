"""Dataset preprocessing for TruthfulQA multiple-choice questions."""

import random
from typing import Dict, List, Any
from collections import Counter
from datasets import load_dataset


def load_truthfulqa(
    split: str = "validation",
    num_samples: int = None,
    seed: int = 42,
    cache_dir: str = ".cache",
) -> List[Dict[str, Any]]:
    """
    Load TruthfulQA multiple-choice dataset.

    Args:
        split: Dataset split (default: validation)
        num_samples: Number of samples to use (None = all)
        seed: Random seed for subsampling
        cache_dir: Cache directory for datasets

    Returns:
        List of dictionaries with question, choices, and correct_answer
    """
    # Load TruthfulQA multiple-choice dataset
    dataset = load_dataset(
        "truthful_qa", "multiple_choice", split=split, cache_dir=cache_dir
    )

    # Convert to list of examples
    examples = []
    for item in dataset:
        # TruthfulQA mc format has mc1_targets (single correct) and mc2_targets (multiple correct/incorrect)
        # We'll use mc1_targets for simplicity
        choices = item.get("mc1_targets", {}).get("choices", [])
        labels = item.get("mc1_targets", {}).get("labels", [])

        if not choices or not labels:
            continue

        # Find correct answer index
        try:
            correct_idx = labels.index(1)
        except ValueError:
            continue

        # Convert to option letters (A, B, C, D, ...)
        choice_letters = [chr(65 + i) for i in range(len(choices))]
        correct_letter = choice_letters[correct_idx]

        examples.append(
            {
                "question": item["question"],
                "choices": choices,
                "choice_letters": choice_letters,
                "correct_answer": correct_letter,
                "correct_idx": correct_idx,
            }
        )

    # [VALIDATOR FIX - Attempt 3]
    # [PROBLEM]: random.shuffle with seed=42 still creates biased subset where all correct answers are "A"
    # [CAUSE]: Even with shuffle, seed=42 happens to select indices that map to examples with "A" answers
    # [FIX]: Use true stratified sampling - group by correct_answer, then sample proportionally from each group
    #
    # [OLD CODE]:
    # random.seed(seed)
    # indices = list(range(len(examples)))
    # random.shuffle(indices)
    # indices = indices[:num_samples]
    # examples = [examples[i] for i in indices]
    #
    # [NEW CODE]:
    # Subsample if requested
    if num_samples is not None and num_samples < len(examples):
        # Stratified sampling: ensure answer diversity by sampling proportionally from each answer group
        random.seed(seed)

        # Group examples by correct answer
        answer_groups = {}
        for ex in examples:
            answer = ex["correct_answer"]
            if answer not in answer_groups:
                answer_groups[answer] = []
            answer_groups[answer].append(ex)

        # Calculate how many to sample from each group (proportional to group size)
        total = len(examples)
        samples_per_group = {}
        for answer, group in answer_groups.items():
            proportion = len(group) / total
            samples_per_group[answer] = max(
                1, int(num_samples * proportion)
            )  # At least 1 from each group

        # Adjust if we overallocated (due to rounding + "at least 1" constraint)
        total_allocated = sum(samples_per_group.values())
        if total_allocated > num_samples:
            # Remove excess from largest groups
            sorted_groups = sorted(
                answer_groups.items(), key=lambda x: len(x[1]), reverse=True
            )
            excess = total_allocated - num_samples
            for answer, _ in sorted_groups:
                if excess == 0:
                    break
                if samples_per_group[answer] > 1:
                    reduction = min(excess, samples_per_group[answer] - 1)
                    samples_per_group[answer] -= reduction
                    excess -= reduction

        # Sample from each group
        sampled = []
        for answer, group in answer_groups.items():
            n = min(samples_per_group[answer], len(group))
            sampled.extend(random.sample(group, n))

        # If we still need more samples (due to rounding), randomly add from remaining
        if len(sampled) < num_samples:
            remaining = num_samples - len(sampled)
            all_remaining = [ex for ex in examples if ex not in sampled]
            if all_remaining:
                sampled.extend(
                    random.sample(all_remaining, min(remaining, len(all_remaining)))
                )

        # Shuffle the final sample to avoid answer clustering
        random.shuffle(sampled)
        examples = sampled[:num_samples]

        # Verify answer diversity
        answer_dist = Counter(ex["correct_answer"] for ex in examples)
        unique_answers = len(answer_dist)
        print(
            f"Sampled {len(examples)} examples with {unique_answers} unique correct answers"
        )
        print(f"  Answer distribution: {dict(answer_dist)}")

        if unique_answers < 2 and len(examples) >= 10:
            print(
                f"WARNING: Only {unique_answers} unique correct answers in {len(examples)} samples!"
            )

    return examples


def format_question(example: Dict[str, Any]) -> str:
    """
    Format a question with answer choices for the model.

    Args:
        example: Dictionary with question, choices, and choice_letters

    Returns:
        Formatted question string
    """
    question = example["question"]
    choices = example["choices"]
    letters = example["choice_letters"]

    formatted = f"Question: {question}\n\nAnswer Choices:\n"
    for letter, choice in zip(letters, choices):
        formatted += f"{letter}. {choice}\n"

    return formatted.strip()
