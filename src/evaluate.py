"""Evaluation script for comparing EC-CoT runs."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import wandb
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend


def fetch_wandb_run(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB by display name.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with config, summary, and history
    """
    api = wandb.Api()

    # Fetch runs with matching display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        raise ValueError(f"No run found with display name: {run_id}")

    run = runs[0]  # Most recent

    # Get history (logged metrics over time)
    history = run.history()

    return {
        "config": dict(run.config),
        "summary": dict(run.summary),
        "history": history,
        "url": run.url,
    }


def export_run_metrics(run_data: Dict[str, Any], output_dir: Path) -> None:
    """
    Export per-run metrics to JSON.

    Args:
        run_data: Run data from WandB
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "summary": run_data["summary"],
        "config": run_data["config"],
        "url": run_data["url"],
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Exported metrics to: {output_dir / 'metrics.json'}")


def create_run_figures(run_data: Dict[str, Any], output_dir: Path, run_id: str) -> None:
    """
    Create per-run visualizations.

    Args:
        run_data: Run data from WandB
        output_dir: Output directory
        run_id: Run identifier
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    history = run_data["history"]

    if "accuracy_running" in history.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(history["example_idx"], history["accuracy_running"])
        plt.xlabel("Example Index")
        plt.ylabel("Running Accuracy")
        plt.title(f"Running Accuracy - {run_id}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = output_dir / "running_accuracy.pdf"
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"  Created figure: {output_path}")


def create_comparison_figures(
    all_runs: Dict[str, Dict[str, Any]], output_dir: Path
) -> None:
    """
    Create comparison figures across runs.

    Args:
        all_runs: Dictionary mapping run_id to run_data
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Running accuracy comparison
    plt.figure(figsize=(12, 7))
    for run_id, run_data in all_runs.items():
        history = run_data["history"]
        if "accuracy_running" in history.columns:
            plt.plot(
                history["example_idx"],
                history["accuracy_running"],
                label=run_id,
                linewidth=2,
            )

    plt.xlabel("Example Index", fontsize=12)
    plt.ylabel("Running Accuracy", fontsize=12)
    plt.title("Running Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "comparison_accuracy.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Created comparison figure: {output_path}")

    # Final metrics bar chart
    metric_keys = ["accuracy", "avg_stability"]

    # Check if coverage is available (EC-CoT specific)
    if any("avg_coverage" in run_data["summary"] for run_data in all_runs.values()):
        metric_keys.append("avg_coverage")

    for metric_key in metric_keys:
        plt.figure(figsize=(10, 6))

        run_ids = []
        values = []

        for run_id, run_data in all_runs.items():
            if metric_key in run_data["summary"]:
                run_ids.append(run_id)
                values.append(run_data["summary"][metric_key])

        if values:
            plt.bar(range(len(run_ids)), values, tick_label=run_ids)
            plt.ylabel(metric_key.replace("_", " ").title(), fontsize=12)
            plt.title(
                f"{metric_key.replace('_', ' ').title()} Comparison",
                fontsize=14,
                fontweight="bold",
            )
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()

            output_path = output_dir / f"comparison_{metric_key}.pdf"
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close()
            print(f"  Created comparison figure: {output_path}")


def export_aggregated_metrics(
    all_runs: Dict[str, Dict[str, Any]], output_dir: Path
) -> None:
    """
    Export aggregated metrics across runs.

    Args:
        all_runs: Dictionary mapping run_id to run_data
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics by run
    metrics_by_run = {}
    for run_id, run_data in all_runs.items():
        metrics_by_run[run_id] = run_data["summary"]

    # Identify primary metric
    primary_metric = "accuracy"

    # Find best proposed and baseline
    proposed_runs = {rid: data for rid, data in all_runs.items() if "proposed" in rid}
    baseline_runs = {
        rid: data for rid, data in all_runs.items() if "comparative" in rid
    }

    best_proposed = None
    best_proposed_value = -1
    if proposed_runs:
        for run_id, run_data in proposed_runs.items():
            value = run_data["summary"].get(primary_metric, 0)
            if value > best_proposed_value:
                best_proposed_value = value
                best_proposed = run_id

    best_baseline = None
    best_baseline_value = -1
    if baseline_runs:
        for run_id, run_data in baseline_runs.items():
            value = run_data["summary"].get(primary_metric, 0)
            if value > best_baseline_value:
                best_baseline_value = value
                best_baseline = run_id

    gap = (
        best_proposed_value - best_baseline_value
        if (best_proposed and best_baseline)
        else None
    )

    aggregated = {
        "primary_metric": primary_metric,
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_proposed_value": best_proposed_value if best_proposed else None,
        "best_baseline": best_baseline,
        "best_baseline_value": best_baseline_value if best_baseline else None,
        "gap": gap,
    }

    with open(output_dir / "aggregated_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"  Exported aggregated metrics to: {output_dir / 'aggregated_metrics.json'}")

    if gap is not None:
        print(f"\n  Best Proposed ({best_proposed}): {best_proposed_value:.4f}")
        print(f"  Best Baseline ({best_baseline}): {best_baseline_value:.4f}")
        print(f"  Gap: {gap:+.4f}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate EC-CoT experiment runs")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON list of run IDs"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (defaults to env or config)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project (defaults to env or config)",
    )

    args = parser.parse_args()

    # Parse run IDs
    run_ids = json.loads(args.run_ids)

    # Get WandB config
    wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY", "airas")
    wandb_project = args.wandb_project or os.environ.get(
        "WANDB_PROJECT", "20260313-tanaka-test"
    )

    print("=" * 80)
    print("EC-CoT Evaluation")
    print(f"Run IDs: {run_ids}")
    print(f"WandB: {wandb_entity}/{wandb_project}")
    print("=" * 80)

    # Fetch all runs
    all_runs = {}
    for run_id in run_ids:
        print(f"\nFetching run: {run_id}")
        try:
            run_data = fetch_wandb_run(wandb_entity, wandb_project, run_id)
            all_runs[run_id] = run_data
            print(f"  URL: {run_data['url']}")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not all_runs:
        print("\nNo runs fetched successfully. Exiting.")
        return

    results_dir = Path(args.results_dir)

    # Export per-run metrics and figures
    print("\n" + "=" * 80)
    print("Exporting per-run metrics and figures...")
    print("=" * 80)

    for run_id, run_data in all_runs.items():
        print(f"\nProcessing run: {run_id}")
        run_output_dir = results_dir / run_id
        export_run_metrics(run_data, run_output_dir)
        create_run_figures(run_data, run_output_dir, run_id)

    # Create comparison figures
    print("\n" + "=" * 80)
    print("Creating comparison figures...")
    print("=" * 80)

    comparison_dir = results_dir / "comparison"
    create_comparison_figures(all_runs, comparison_dir)

    # Export aggregated metrics
    print("\n" + "=" * 80)
    print("Exporting aggregated metrics...")
    print("=" * 80)

    export_aggregated_metrics(all_runs, comparison_dir)

    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
