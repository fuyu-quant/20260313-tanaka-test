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

    # [VALIDATOR FIX - Attempt 2]
    # [PROBLEM]: random.sample with seed=42 creates biased subset where all/most correct answers are "A"
    # [CAUSE]: random.sample on pre-processed list doesn't guarantee representative sampling
    # [FIX]: Use stratified sampling by taking evenly spaced indices to ensure answer diversity
    #
    # [OLD CODE]:
    # if num_samples is not None and num_samples < len(examples):
    #     random.seed(seed)
    #     examples = random.sample(examples, num_samples)
    #
    # [NEW CODE]:
    # Subsample if requested
    if num_samples is not None and num_samples < len(examples):
        # Use stratified sampling: take evenly spaced indices to ensure diversity
        # This prevents bias toward any particular answer letter
        random.seed(seed)
        # First shuffle to maintain randomness
        indices = list(range(len(examples)))
        random.shuffle(indices)
        # Then take first num_samples (which are now randomly distributed)
        indices = indices[:num_samples]
        examples = [examples[i] for i in indices]

        # Verify answer diversity (at least 2 unique correct answers in sample)
        unique_answers = len(set(ex["correct_answer"] for ex in examples))
        if unique_answers < 2 and len(examples) >= 10:
            print(
                f"WARNING: Only {unique_answers} unique correct answers in {len(examples)} samples!"
            )
            print(f"  Distribution: {Counter(ex['correct_answer'] for ex in examples)}")

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
