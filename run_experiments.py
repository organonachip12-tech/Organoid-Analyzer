#!/usr/bin/env python3
"""
Ultra-Simple Experiment Runner
Just runs the scripts with different parameters - no complexity!
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
import seaborn as sns

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich import box


def load_config(config_path):
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)

def check_experiment_exists(experiment_dir, analyzer_type):
    """Check if experiment results already exist."""
    if not os.path.exists(experiment_dir):
        return False
    
    if analyzer_type == "organoid":
        # Check for CSV files in accuracies directory (including subdirectories)
        accuracies_dir = os.path.join(experiment_dir, "ablation_Specify", "accuracies")
        if os.path.exists(accuracies_dir):
            # Check subdirectories for CSV files
            for subdir in os.listdir(accuracies_dir):
                subdir_path = os.path.join(accuracies_dir, subdir)
                if os.path.isdir(subdir_path):
                    csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
                    if csv_files:
                        return True
    
    elif analyzer_type == "til":
        # Check for Excel files in metrics directory
        metrics_dir = os.path.join(experiment_dir, "metrics")
        if os.path.exists(metrics_dir):
            xlsx_files = [f for f in os.listdir(metrics_dir) if f.endswith('.xlsx')]
            if xlsx_files:
                return True
    
    return False

def calculate_survival_metrics(df):
    """Calculate meaningful survival analysis metrics from TIL data."""
    metrics = {}
    
    try:
        # Basic data info
        metrics['n_samples'] = len(df)
        metrics['n_timepoints'] = len(df.columns) - 1  # Exclude first column (likely sample ID)
        
        # Convert to numeric, handling mixed data types
        survival_data = df.iloc[:, 1:]  # Exclude first column
        
        # Convert all data to numeric, coercing errors to NaN
        survival_data_numeric = survival_data.apply(pd.to_numeric, errors='coerce')
        
        # Remove columns that are all NaN (likely non-numeric columns)
        survival_data_numeric = survival_data_numeric.dropna(axis=1, how='all')
        
        if survival_data_numeric.empty:
            metrics['error'] = "No numeric data found in survival columns"
            return metrics
        
        # Mean survival probability at different timepoints
        mean_survival = survival_data_numeric.mean()
        
        # Calculate metrics based on available timepoints
        n_timepoints = len(mean_survival)
        
        # Use proportional indices for timepoints
        idx_1yr = min(7, n_timepoints - 1) if n_timepoints > 0 else 0
        idx_2yr = min(14, n_timepoints - 1) if n_timepoints > 0 else 0
        idx_final = n_timepoints - 1 if n_timepoints > 0 else 0
        
        metrics['mean_survival_1yr'] = float(mean_survival.iloc[idx_1yr]) if n_timepoints > 0 else None
        metrics['mean_survival_2yr'] = float(mean_survival.iloc[idx_2yr]) if n_timepoints > 0 else None
        metrics['mean_survival_final'] = float(mean_survival.iloc[idx_final]) if n_timepoints > 0 else None
        
        # Survival curve variance (lower = more consistent predictions)
        survival_var = survival_data_numeric.var()
        metrics['survival_variance'] = float(survival_var.mean()) if not survival_var.empty else 0
        
        # Calculate survival curve smoothness (lower = smoother curves)
        # This measures how much the survival curves change between timepoints
        diff_matrix = survival_data_numeric.diff(axis=1).abs()
        metrics['curve_smoothness'] = float(diff_matrix.mean().mean()) if not diff_matrix.empty else 0
        
        # Calculate survival curve discrimination
        # Higher values indicate better separation between high/low risk groups
        survival_std = survival_data_numeric.std()
        metrics['survival_discrimination'] = float(survival_std.mean()) if not survival_std.empty else 0
        
        # Calculate overall model confidence (inverse of variance)
        metrics['model_confidence'] = 1.0 / (metrics['survival_variance'] + 1e-8)
        
        # Calculate survival curve quality score (composite metric)
        # Combines smoothness, discrimination, and confidence
        quality_score = (
            (1.0 / (metrics['curve_smoothness'] + 1e-8)) * 0.3 +  # Smoothness (30%)
            metrics['survival_discrimination'] * 0.4 +  # Discrimination (40%)
            metrics['model_confidence'] * 0.3  # Confidence (30%)
        )
        metrics['quality_score'] = quality_score
        
    except Exception as e:
        metrics['error'] = str(e)
        import traceback
        metrics['traceback'] = traceback.format_exc()
    
    return metrics

def analyze_results(results_dir):
    """Analyze all experiment results and generate summary."""
    console = Console()
    
    console.print("\n" + "="*80)
    console.print("📊 EXPERIMENT RESULTS ANALYSIS", style="bold blue")
    console.print("="*80)
    
    # Find all experiment directories
    experiment_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            experiment_dirs.append(item_path)
    
    if not experiment_dirs:
        console.print("❌ No experiment results found!", style="red")
        return
    
    # Analyze Organoid results
    organoid_results = []
    til_results = []
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        
        if exp_name.startswith("organoid_"):
            # Analyze Organoid CSV files
            accuracies_dir = os.path.join(exp_dir, "ablation_Specify", "accuracies")
            if os.path.exists(accuracies_dir):
                # Look for CSV files in subdirectories
                latest_csv_path = None
                latest_time = 0
                
                for subdir in os.listdir(accuracies_dir):
                    subdir_path = os.path.join(accuracies_dir, subdir)
                    if os.path.isdir(subdir_path):
                        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
                        for csv_file in csv_files:
                            csv_path = os.path.join(subdir_path, csv_file)
                            file_time = os.path.getmtime(csv_path)
                            if file_time > latest_time:
                                latest_time = file_time
                                latest_csv_path = csv_path
                
                if latest_csv_path:
                    try:
                        df = pd.read_csv(latest_csv_path, header=None)
                        # Extract test accuracy (last row, second column)
                        test_acc = float(df.iloc[-1, 1]) if len(df) > 1 else 0
                        organoid_results.append({
                            'experiment': exp_name,
                            'test_accuracy': test_acc,
                            'file': latest_csv_path
                        })
                    except Exception as e:
                        console.print(f"⚠️  Could not read {latest_csv_path}: {e}", style="yellow")
        
        elif exp_name.startswith("til_"):
            # Analyze TIL Excel files
            metrics_dir = os.path.join(exp_dir, "metrics")
            if os.path.exists(metrics_dir):
                xlsx_files = [f for f in os.listdir(metrics_dir) if f.endswith('.xlsx')]
                if xlsx_files:
                    latest_xlsx = sorted(xlsx_files)[-1]  # Get latest file
                    xlsx_path = os.path.join(metrics_dir, latest_xlsx)
                    try:
                        df = pd.read_excel(xlsx_path)
                        
                        # Calculate survival analysis metrics
                        metrics = calculate_survival_metrics(df)
                        
                        til_results.append({
                            'experiment': exp_name,
                            'file': xlsx_path,
                            'metrics': metrics
                        })
                    except Exception as e:
                        console.print(f"⚠️  Could not read {xlsx_path}: {e}", style="yellow")
    
    # Display results summary
    if organoid_results:
        console.print(f"\n🧬 ORGANOID ANALYZER RESULTS ({len(organoid_results)} experiments)", style="bold green")
        
        table = Table(title="Organoid Experiment Results", box=box.ROUNDED)
        table.add_column("Experiment", style="cyan")
        table.add_column("Test Accuracy", style="magenta", justify="right")
        
        best_acc = 0
        best_exp = None
        
        for result in organoid_results:
            acc = result['test_accuracy']
            table.add_row(result['experiment'], f"{acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_exp = result['experiment']
        
        console.print(table)
        
        if best_exp:
            console.print(f"\n🏆 Best Organoid Performance: {best_exp} (Accuracy: {best_acc:.4f})", style="bold green")
    
    if til_results:
        console.print(f"\n🔬 TIL ANALYZER RESULTS ({len(til_results)} experiments)", style="bold blue")
        
        table = Table(title="TIL Experiment Results", box=box.ROUNDED)
        table.add_column("Experiment", style="cyan")
        table.add_column("Samples", style="magenta", justify="right")
        table.add_column("1yr Survival", style="green", justify="right")
        table.add_column("2yr Survival", style="green", justify="right")
        table.add_column("Quality Score", style="yellow", justify="right")
        table.add_column("Discrimination", style="blue", justify="right")
        
        best_quality = 0
        best_exp = None
        
        for result in til_results:
            metrics = result['metrics']
            if 'error' not in metrics:
                table.add_row(
                    result['experiment'],
                    str(metrics['n_samples']),
                    f"{metrics['mean_survival_1yr']:.3f}" if metrics['mean_survival_1yr'] else "N/A",
                    f"{metrics['mean_survival_2yr']:.3f}" if metrics['mean_survival_2yr'] else "N/A",
                    f"{metrics['quality_score']:.3f}",
                    f"{metrics['survival_discrimination']:.3f}"
                )
                
                if metrics['quality_score'] > best_quality:
                    best_quality = metrics['quality_score']
                    best_exp = result['experiment']
            else:
                table.add_row(
                    result['experiment'],
                    "Error",
                    "N/A", "N/A", "N/A", "N/A"
                )
        
        console.print(table)
        
        if best_exp:
            console.print(f"\n🏆 Best TIL Performance: {best_exp} (Quality Score: {best_quality:.3f})", style="bold blue")
    
    # Generate summary plots
    if organoid_results or til_results:
        try:
            n_plots = 0
            if organoid_results:
                n_plots += 2
            if til_results:
                n_plots += 2
            
            plt.figure(figsize=(16, 8))
            plot_idx = 1
            
            if organoid_results:
                # Plot 1: Organoid Test Accuracies
                plt.subplot(2, 2, plot_idx)
                exp_names = [r['experiment'].replace('organoid_', '') for r in organoid_results]
                accuracies = [r['test_accuracy'] for r in organoid_results]
                
                bars = plt.bar(range(len(exp_names)), accuracies, color='skyblue', alpha=0.7)
                plt.xlabel('Experiment')
                plt.ylabel('Test Accuracy')
                plt.title('Organoid Analyzer - Test Accuracies')
                plt.xticks(range(len(exp_names)), exp_names, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{acc:.3f}', ha='center', va='bottom')
                plot_idx += 1
                
                # Plot 2: Organoid Accuracy distribution
                plt.subplot(2, 2, plot_idx)
                plt.hist(accuracies, bins=min(10, len(accuracies)), color='lightcoral', alpha=0.7, edgecolor='black')
                plt.xlabel('Test Accuracy')
                plt.ylabel('Frequency')
                plt.title('Organoid Accuracy Distribution')
                plt.grid(True, alpha=0.3)
                plot_idx += 1
            
            if til_results:
                # Plot 3: TIL Quality Scores
                plt.subplot(2, 2, plot_idx)
                til_exp_names = [r['experiment'].replace('til_', '') for r in til_results]
                quality_scores = [r['metrics'].get('quality_score', 0) for r in til_results if 'error' not in r['metrics']]
                
                if quality_scores:
                    bars = plt.bar(range(len(til_exp_names)), quality_scores, color='lightgreen', alpha=0.7)
                    plt.xlabel('Experiment')
                    plt.ylabel('Quality Score')
                    plt.title('TIL Analyzer - Quality Scores')
                    plt.xticks(range(len(til_exp_names)), til_exp_names, rotation=45, ha='right')
                    plt.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, score in zip(bars, quality_scores):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                f'{score:.3f}', ha='center', va='bottom')
                plot_idx += 1
                
                # Plot 4: TIL Survival Discrimination
                plt.subplot(2, 2, plot_idx)
                discrimination_scores = [r['metrics'].get('survival_discrimination', 0) for r in til_results if 'error' not in r['metrics']]
                
                if discrimination_scores:
                    plt.hist(discrimination_scores, bins=min(10, len(discrimination_scores)), color='lightblue', alpha=0.7, edgecolor='black')
                    plt.xlabel('Survival Discrimination')
                    plt.ylabel('Frequency')
                    plt.title('TIL Discrimination Distribution')
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(results_dir, 'experiment_summary.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            console.print(f"\n📈 Summary plot saved: {plot_path}", style="green")
            
        except Exception as e:
            console.print(f"⚠️  Could not generate plots: {e}", style="yellow")
    
    console.print(f"\n✅ Analysis complete! Found {len(organoid_results)} Organoid and {len(til_results)} TIL experiments.", style="bold green")


def generate_combinations(experiment_config, test_mode=False):
    """Generate parameter combinations."""
    param_lists = {k: v for k, v in experiment_config.items() 
                  if k != "name" and isinstance(v, list)}
    
    param_names = list(param_lists.keys())
    param_values = list(param_lists.values())
    
    combinations = []
    for combo in itertools.product(*param_values):
        combination = dict(zip(param_names, combo))
        combinations.append(combination)
        
        if test_mode:
            break  # Only first combination in test mode
    
    return combinations


def run_organoid_experiment(combination, output_dir):
    """Run Organoid experiment with given parameters."""
    cmd = [
        "python", "train_and_test_models.py",
        "--seq_len", str(combination.get('seq_len', 100)),
        "--epochs", str(combination.get('epochs', 400)),
        "--batch_size", str(combination.get('batch_size', 256)),
        "--dropout", str(combination.get('dropout', 0.3)),
        "--hidden_sizes", str(combination.get('hidden_sizes', 32)),
        "--fusion_sizes", str(combination.get('fusion_sizes', 64)),
        "--output_dir", str(output_dir)
    ]
    
    print(f"🧬 Running Organoid with: {combination}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Organoid experiment completed successfully")
        return True
    else:
        print(f"❌ Organoid experiment failed:")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
        return False


def run_til_experiment(combination, output_dir):
    """Run TIL experiment with given parameters."""
    cmd = [
        "python", "main_test.py",
        "--intermediate_features", str(combination.get('intermediate_features', 512)),
        "--second_intermediate_features", str(combination.get('second_intermediate_features', 512)),
        "--dropout_prob", str(combination.get('dropout_prob', 0.5)),
        "--batch_size", str(combination.get('batch_size', 16)),
        "--epochs", str(combination.get('epochs', 200)),
        "--learning_rate", str(combination.get('learning_rate', 0.00025)),
        "--output_dir", str(output_dir)
    ]
    
    print(f"🔬 Running TIL with: {combination}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ TIL experiment completed successfully")
        return True
    else:
        print(f"❌ TIL experiment failed:")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
        return False


def main():
    """Main function - ultra simple!"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Simple Experiment Runner")
    parser.add_argument("--config", default="experiments_config.json", help="Config file")
    parser.add_argument("--analyzer", choices=["organoid", "til", "all"], default="all", help="Which analyzer")
    parser.add_argument("--test", action="store_true", help="Test mode: only 1 combination each")
    
    args = parser.parse_args()
    
    console = Console()
    
    # Load config
    config = load_config(args.config)
    
    # Show what we're doing
    test_text = "🧪 TEST MODE - Only 1 combination per experiment" if args.test else ""
    console.print(Panel.fit(
        f"🎯 [bold blue]Ultra-Simple Experiment Runner[/bold blue]\n"
        f"📁 Config: [cyan]{args.config}[/cyan]\n"
        f"{test_text}",
        title="🚀 Starting",
        border_style="blue"
    ))
    
    completed = 0
    failed = 0
    total = 0
    
    # Run Organoid experiments
    if args.analyzer in ["organoid", "all"]:
        console.print("\n🧬 [bold green]Running Organoid experiments...[/bold green]")
        
        original_cwd = os.getcwd()
        try:
            os.chdir("Organoid Analyzer")
            
            # Collect all experiments first
            all_experiments = []
            for experiment_config in config["organoid_experiments"]:
                combinations = generate_combinations(experiment_config, args.test)
                for i, combination in enumerate(combinations):
                    experiment_id = f"organoid_{experiment_config['name']}_{i+1}"
                    output_dir = f"../experiments_results/{experiment_id}"
                    all_experiments.append((experiment_id, combination, output_dir))
            
            # Filter out completed experiments
            experiments_to_run = []
            skipped_count = 0
            
            for experiment_id, combination, output_dir in all_experiments:
                if check_experiment_exists(output_dir, "organoid"):
                    console.print(f"⏭️  Skipping {experiment_id} (results already exist)", style="yellow")
                    skipped_count += 1
                else:
                    experiments_to_run.append((experiment_id, combination, output_dir))
            
            if skipped_count > 0:
                console.print(f"📋 Skipped {skipped_count} completed Organoid experiments", style="cyan")
            
            if experiments_to_run:
                # Set up progress tracking
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    
                    task = progress.add_task("Running Organoid experiments", total=len(experiments_to_run))
                    
                    for experiment_id, combination, output_dir in experiments_to_run:
                        progress.update(task, description=f"Running {experiment_id}")
                        
                        success = run_organoid_experiment(combination, output_dir)
                        if success:
                            completed += 1
                        else:
                            failed += 1
                        
                        total += 1
                        progress.advance(task)
            else:
                console.print("✅ All Organoid experiments already completed!", style="green")
                        
        finally:
            os.chdir(original_cwd)
    
    # Run TIL experiments  
    if args.analyzer in ["til", "all"]:
        console.print("\n🔬 [bold green]Running TIL experiments...[/bold green]")
        
        original_cwd = os.getcwd()
        try:
            os.chdir("TIL-Analyzer-main")
            
            # Collect all experiments first
            all_experiments = []
            for experiment_config in config["til_experiments"]:
                combinations = generate_combinations(experiment_config, args.test)
                for i, combination in enumerate(combinations):
                    experiment_id = f"til_{experiment_config['name']}_{i+1}"
                    output_dir = f"../experiments_results/{experiment_id}"
                    all_experiments.append((experiment_id, combination, output_dir))
            
            # Filter out completed experiments
            experiments_to_run = []
            skipped_count = 0
            
            for experiment_id, combination, output_dir in all_experiments:
                if check_experiment_exists(output_dir, "til"):
                    console.print(f"⏭️  Skipping {experiment_id} (results already exist)", style="yellow")
                    skipped_count += 1
                else:
                    experiments_to_run.append((experiment_id, combination, output_dir))
            
            if skipped_count > 0:
                console.print(f"📋 Skipped {skipped_count} completed TIL experiments", style="cyan")
            
            if experiments_to_run:
                # Set up progress tracking
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    
                    task = progress.add_task("Running TIL experiments", total=len(experiments_to_run))
                    
                    for experiment_id, combination, output_dir in experiments_to_run:
                        progress.update(task, description=f"Running {experiment_id}")
                        
                        success = run_til_experiment(combination, output_dir)
                        if success:
                            completed += 1
                        else:
                            failed += 1
                        
                        total += 1
                        progress.advance(task)
            else:
                console.print("✅ All TIL experiments already completed!", style="green")
                        
        finally:
            os.chdir(original_cwd)
    
    # Show results
    console.print(Panel.fit(
        f"🎉 [bold green]All done![/bold green]\n"
        f"✅ Completed: [green]{completed}[/green]\n"
        f"❌ Failed: [red]{failed}[/red]\n"
        f"📊 Total: [blue]{total}[/blue]",
        title="🎉 Results",
        border_style="green"
    ))
    
    # Always analyze results (including previously completed experiments)
    results_dir = config["global_settings"]["output_dir"]
    if os.path.exists(results_dir):
        analyze_results(results_dir)
    else:
        console.print("📊 No results directory found - no experiments have been run yet.", style="yellow")


if __name__ == "__main__":
    main()