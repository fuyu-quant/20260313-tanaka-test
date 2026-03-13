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

    # [VALIDATOR FIX - Attempt 4]
    # [PROBLEM]: All examples have correct_answer="A" because TruthfulQA always puts correct answer first
    # [CAUSE]: TruthfulQA mc1_targets format has labels=[1,0,0,...] with correct answer always at index 0
    # [FIX]: Shuffle choices before assigning letters to ensure answer diversity
    #
    # [OLD CODE]:
    # choice_letters = [chr(65 + i) for i in range(len(choices))]
    # correct_letter = choice_letters[correct_idx]
    #
    # [NEW CODE]:
    # Convert to list of examples
    examples = []
    rng = random.Random(seed)  # Separate RNG for shuffling choices (deterministic)

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

        # Shuffle choices to randomize answer positions (fix: TruthfulQA always puts correct answer first)
        # Create paired list of (choice, is_correct) and shuffle
        paired = list(zip(choices, labels))
        rng.shuffle(paired)
        shuffled_choices, shuffled_labels = zip(*paired)

        # Find new correct answer index after shuffling
        correct_idx = shuffled_labels.index(1)

        # Convert to option letters (A, B, C, D, ...)
        choice_letters = [chr(65 + i) for i in range(len(shuffled_choices))]
        correct_letter = choice_letters[correct_idx]

        examples.append(
            {
                "question": item["question"],
                "choices": list(shuffled_choices),
                "choice_letters": choice_letters,
                "correct_answer": correct_letter,
                "correct_idx": correct_idx,
            }
        )

    # [VALIDATOR FIX - Attempt 4]
    # [PROBLEM]: Stratified sampling with "at least 1 from each group" fails when num_samples < num_groups
    # [CAUSE]: With 10 samples and potentially 20+ answer letters, "at least 1 from each" is impossible
    # [FIX]: Guarantee diversity by sampling at least 2 different answers, then fill the rest randomly
    #
    # [OLD CODE]:
    # (complex stratified sampling with "at least 1 from each group" constraint)
    #
    # [NEW CODE]:
    # Subsample if requested
    if num_samples is not None and num_samples < len(examples):
        random.seed(seed)

        # Group examples by correct answer
        answer_groups = {}
        for ex in examples:
            answer = ex["correct_answer"]
            if answer not in answer_groups:
                answer_groups[answer] = []
            answer_groups[answer].append(ex)

        num_unique_answers = len(answer_groups)

        # Strategy: For sanity checks, we need answer diversity
        # Sample at least min(num_samples // 2, num_unique_answers) different answer types
        # This ensures diversity without over-constraining
        sampled = []

        if num_samples >= 4 and num_unique_answers >= 2:
            # Ensure diversity: pick at least 2-3 different answer types
            min_answer_types = min(max(2, num_samples // 5), num_unique_answers)

            # Randomly select which answer types to include
            selected_answers = random.sample(
                list(answer_groups.keys()), min_answer_types
            )

            # Allocate samples proportionally, ensuring at least 1 from each selected type
            samples_per_type = num_samples // min_answer_types
            remainder = num_samples % min_answer_types

            for i, answer in enumerate(selected_answers):
                # Give extra samples to first few groups (to use up remainder)
                n = samples_per_type + (1 if i < remainder else 0)
                n = min(n, len(answer_groups[answer]))  # Don't exceed group size
                sampled.extend(random.sample(answer_groups[answer], n))

            # If we're short, add more randomly from remaining examples
            if len(sampled) < num_samples:
                remaining = num_samples - len(sampled)
                all_remaining = [ex for ex in examples if ex not in sampled]
                if all_remaining:
                    sampled.extend(
                        random.sample(all_remaining, min(remaining, len(all_remaining)))
                    )

            # Shuffle to avoid answer clustering
            random.shuffle(sampled)
            examples = sampled[:num_samples]
        else:
            # For very small samples or limited diversity, use simple random sampling
            examples = random.sample(examples, min(num_samples, len(examples)))

        # Verify answer diversity
        answer_dist = Counter(ex["correct_answer"] for ex in examples)
        unique_answers = len(answer_dist)
        print(
            f"Sampled {len(examples)} examples with {unique_answers} unique correct answers (total available: {num_unique_answers})"
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
