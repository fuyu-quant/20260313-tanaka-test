"""Main orchestrator for EC-CoT experiments."""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.inference import run_inference


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running a single experiment.

    Handles mode overrides and delegates to appropriate runner.

    Args:
        cfg: Hydra configuration
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: ConfigAttributeError: Key 'method' is not in struct
    # [CAUSE]: Hydra loads run configs under 'run' group, but code tried to access cfg.method directly
    # [FIX]: Access all run config fields through cfg.run namespace
    #
    # [OLD CODE]:
    # print(f"Method: {cfg.method.type}")
    # cfg.dataset.num_samples = min(10, cfg.dataset.num_samples)
    #
    # [NEW CODE]:
    print("=" * 80)
    print(f"EC-CoT Experiment Runner")
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.type}")
    print("=" * 80)

    # [VALIDATOR FIX - Attempt 2]
    # [PROBLEM]: Mode was "sanity_check" but code only checked for "sanity", so validation didn't run
    # [CAUSE]: GitHub Actions passes "sanity_check" as mode, but code expected "sanity"
    # [FIX]: Normalize mode to handle both "sanity" and "sanity_check" variants
    #
    # [OLD CODE]:
    # if cfg.mode == "sanity":
    #
    # [NEW CODE]:
    # Normalize mode names
    mode = cfg.mode.replace("_check", "").replace("-check", "")
    if mode == "sanity":
        cfg.mode = "sanity"  # Normalize for downstream checks

    # Apply mode-specific overrides
    if mode == "sanity":
        print("Applying sanity mode overrides...")
        # Reduce samples for quick validation
        cfg.run.dataset.num_samples = min(10, cfg.run.dataset.num_samples)
        # Set wandb project to avoid polluting full runs
        if not cfg.wandb.project.endswith("-sanity"):
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"
        print(f"  - num_samples: {cfg.run.dataset.num_samples}")
        print(f"  - wandb.project: {cfg.wandb.project}")

    elif mode == "pilot":
        print("Applying pilot mode overrides...")
        # Use 20% of samples (at least 50)
        original_samples = cfg.run.dataset.num_samples
        cfg.run.dataset.num_samples = max(50, int(original_samples * 0.2))
        # Set wandb project to avoid polluting full runs
        if not cfg.wandb.project.endswith("-pilot"):
            cfg.wandb.project = f"{cfg.wandb.project}-pilot"
        print(
            f"  - num_samples: {cfg.run.dataset.num_samples} (20% of {original_samples})"
        )
        print(f"  - wandb.project: {cfg.wandb.project}")

    # Print final config
    print("\nFinal Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Run inference (this is an inference-only task)
    run_inference(cfg)

    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
