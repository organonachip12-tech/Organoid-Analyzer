#!/usr/bin/env python3
"""
Ultra-Simple Experiment Runner (fixed for modular layout)
"""

import json
import os
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
import uuid
import pandas as pd
import matplotlib.pyplot as plt

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich import box
import seaborn as sns
import sys


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def check_experiment_exists(experiment_dir, analyzer_type):
    if not os.path.exists(experiment_dir):
        return False

    if analyzer_type == "organoid":
        accuracies_dir = os.path.join(experiment_dir, "ablation_Specify", "accuracies")
        if os.path.exists(accuracies_dir):
            for subdir in os.listdir(accuracies_dir):
                subdir_path = os.path.join(accuracies_dir, subdir)
                if os.path.isdir(subdir_path):
                    csv_files = [f for f in os.listdir(subdir_path) if f.endswith(".csv")]
                    if csv_files:
                        return True

    elif analyzer_type == "til":
        metrics_dir = os.path.join(experiment_dir, "metrics")
        if os.path.exists(metrics_dir):
            xlsx_files = [f for f in os.listdir(metrics_dir) if f.endswith(".xlsx")]
            if xlsx_files:
                return True

    return False


# --- survival metrics + analyze_results unchanged from your version ---
# (You can paste your full calculate_survival_metrics(...) and analyze_results(...)
# here exactly as they are, no path changes needed.)

# For brevity, I’ll assume you paste your long functions calculate_survival_metrics()
# and analyze_results() unchanged here.


def generate_combinations(experiment_config, test_mode=False):
    param_lists = {
        k: v for k, v in experiment_config.items() if k != "name" and isinstance(v, list)
    }
    param_names = list(param_lists.keys())
    param_values = list(param_lists.values())

    combinations = []
    for combo in itertools.product(*param_values):
        combination = dict(zip(param_names, combo))
        combinations.append(combination)
        if test_mode:
            break
    return combinations


def run_organoid_experiment(combination, output_dir):
    cmd = [
        sys.executable,
        "-m",
        "organoid_analyzer.training.train_and_test_models",
        "--seq_len",
        str(combination.get("seq_len", 100)),
        "--epochs",
        str(combination.get("epochs", 400)),
        "--batch_size",
        str(combination.get("batch_size", 256)),
        "--dropout",
        str(combination.get("dropout", 0.3)),
        "--hidden_sizes",
        str(combination.get("hidden_sizes", 32)),
        "--fusion_sizes",
        str(combination.get("fusion_sizes", 64)),
        "--output_dir",
        str(output_dir),
    ]

    print(f"🧬 Running Organoid with: {combination}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Organoid experiment completed successfully")
        return True
    else:
        print("❌ Organoid experiment failed:")
        print("   stdout:", result.stdout)
        print("   stderr:", result.stderr)
        return False


def run_til_experiment(combination, output_dir):
    cmd = [
        sys.executable,
        "-m",
        "til_analyzer.main_test",
        "--intermediate_features",
        str(combination.get("intermediate_features", 512)),
        "--second_intermediate_features",
        str(combination.get("second_intermediate_features", 512)),
        "--dropout_prob",
        str(combination.get("dropout_prob", 0.5)),
        "--batch_size",
        str(combination.get("batch_size", 16)),
        "--epochs",
        str(combination.get("epochs", 200)),
        "--learning_rate",
        str(combination.get("learning_rate", 0.00025)),
        "--output_dir",
        str(output_dir),
    ]

    print(f"🔬 Running TIL with: {combination}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ TIL experiment completed successfully")
        return True
    else:
        print("❌ TIL experiment failed:")
        print("   stdout:", result.stdout)
        print("   stderr:", result.stderr)
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ultra-Simple Experiment Runner")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("experiments_config.json")),
        help="Config file",
    )
    parser.add_argument(
        "--analyzer",
        choices=["organoid", "til", "all"],
        default="all",
        help="Which analyzer",
    )
    parser.add_argument("--test", action="store_true", help="Test mode: only 1 combo")

    args = parser.parse_args()
    console = Console()

    config = load_config(args.config)
    test_text = "🧪 TEST MODE - Only 1 combination per experiment" if args.test else ""

    console.print(
        Panel.fit(
            f"🎯 [bold blue]Ultra-Simple Experiment Runner[/bold blue]\n"
            f"📁 Config: [cyan]{args.config}[/cyan]\n"
            f"{test_text}",
            title="🚀 Starting",
            border_style="blue",
        )
    )

    completed = 0
    failed = 0
    total = 0

    # --- Organoid experiments ---
    if args.analyzer in ["organoid", "all"]:
        console.print("\n🧬 [bold green]Running Organoid experiments...[/bold green]")
        all_experiments = []

        for experiment_config in config.get("organoid_experiments", []):
            combinations = generate_combinations(experiment_config, args.test)
            for i, combo in enumerate(combinations):
                experiment_id = f"organoid_{experiment_config['name']}_{i+1}"
                output_dir = os.path.join(
                    config["global_settings"]["output_dir"], experiment_id
                )
                all_experiments.append((experiment_id, combo, output_dir))

        experiments_to_run = []
        skipped_count = 0
        for experiment_id, combo, output_dir in all_experiments:
            if check_experiment_exists(output_dir, "organoid"):
                console.print(
                    f"⏭️  Skipping {experiment_id} (results already exist)", style="yellow"
                )
                skipped_count += 1
            else:
                experiments_to_run.append((experiment_id, combo, output_dir))

        if skipped_count > 0:
            console.print(
                f"📋 Skipped {skipped_count} completed Organoid experiments", style="cyan"
            )

        if experiments_to_run:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Running Organoid experiments", total=len(experiments_to_run)
                )

                for experiment_id, combo, output_dir in experiments_to_run:
                    progress.update(task, description=f"Running {experiment_id}")
                    success = run_organoid_experiment(combo, output_dir)
                    if success:
                        completed += 1
                    else:
                        failed += 1
                    total += 1
                    progress.advance(task)
        else:
            console.print("✅ All Organoid experiments already completed!", style="green")

    # --- TIL experiments ---
    if args.analyzer in ["til", "all"]:
        console.print("\n🔬 [bold green]Running TIL experiments...[/bold green]")
        all_experiments = []

        for experiment_config in config.get("til_experiments", []):
            combinations = generate_combinations(experiment_config, args.test)
            for i, combo in enumerate(combinations):
                experiment_id = f"til_{experiment_config['name']}_{i+1}"
                output_dir = os.path.join(
                    config["global_settings"]["output_dir"], experiment_id
                )
                all_experiments.append((experiment_id, combo, output_dir))

        experiments_to_run = []
        skipped_count = 0
        for experiment_id, combo, output_dir in all_experiments:
            if check_experiment_exists(output_dir, "til"):
                console.print(
                    f"⏭️  Skipping {experiment_id} (results already exist)", style="yellow"
                )
                skipped_count += 1
            else:
                experiments_to_run.append((experiment_id, combo, output_dir))

        if skipped_count > 0:
            console.print(
                f"📋 Skipped {skipped_count} completed TIL experiments", style="cyan"
            )

        if experiments_to_run:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Running TIL experiments", total=len(experiments_to_run)
                )

                for experiment_id, combo, output_dir in experiments_to_run:
                    progress.update(task, description=f"Running {experiment_id}")
                    success = run_til_experiment(combo, output_dir)
                    if success:
                        completed += 1
                    else:
                        failed += 1
                    total += 1
                    progress.advance(task)
        else:
            console.print("✅ All TIL experiments already completed!", style="green")

    console.print(
        Panel.fit(
            f"🎉 [bold green]All done![/bold green]\n"
            f"✅ Completed: [green]{completed}[/green]\n"
            f"❌ Failed: [red]{failed}[/red]\n"
            f"📊 Total: [blue]{total}[/blue]",
            title="🎉 Results",
            border_style="green",
        )
    )

    results_dir = config["global_settings"]["output_dir"]
    if os.path.exists(results_dir):
        from .run_experiments_analysis import analyze_results  # if you want to factor it out
        analyze_results(results_dir)
    else:
        console.print(
            "📊 No results directory found - no experiments have been run yet.", style="yellow"
        )


if __name__ == "__main__":
    main()
