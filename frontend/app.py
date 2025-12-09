#!/usr/bin/env python3
"""
Flask Web Frontend for Experiment Management
"""

import json
import os
import subprocess
import threading
import time
import itertools
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Get paths relative to this file's location
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FRONTEND_DIR)

app = Flask(__name__, 
            template_folder=os.path.join(FRONTEND_DIR, 'templates'),
            static_folder=os.path.join(FRONTEND_DIR, 'static'))
CORS(app)

# Global state for running experiments
running_experiments = {}
experiment_status = {}

CONFIG_PATH = os.path.join(FRONTEND_DIR, "experiments_config.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")

def load_config():
    """Load experiment configuration."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            # Remove any error key that might have been saved incorrectly
            if "error" in config:
                del config["error"]
            return config
    except FileNotFoundError:
        # Return a valid default config instead of an error object
        default_config = {
            "global_settings": {
                "max_parallel_experiments": 2,
                "output_dir": os.path.join(PROJECT_ROOT, "results", "experiments"),
                "log_level": "INFO"
            },
            "organoid_experiments": [],
            "til_experiments": []
        }
        # Save the default config so it exists for next time
        save_config(default_config)
        return default_config

def save_config(config):
    """Save experiment configuration."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

def check_experiment_exists(experiment_dir, analyzer_type):
    """Check if experiment results already exist."""
    if not os.path.exists(experiment_dir):
        return False
    
    if analyzer_type == "organoid":
        accuracies_dir = os.path.join(experiment_dir, "ablation_Specify", "accuracies")
        if os.path.exists(accuracies_dir):
            for subdir in os.listdir(accuracies_dir):
                subdir_path = os.path.join(accuracies_dir, subdir)
                if os.path.isdir(subdir_path):
                    csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
                    if csv_files:
                        return True
    
    elif analyzer_type == "til":
        metrics_dir = os.path.join(experiment_dir, "metrics")
        if os.path.exists(metrics_dir):
            xlsx_files = [f for f in os.listdir(metrics_dir) if f.endswith('.xlsx')]
            if xlsx_files:
                return True
    
    return False

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
            break
    
    return combinations

def run_experiments_async(analyzer_type, test_mode, job_id):
    """Run experiments in background thread."""
    global experiment_status
    
    try:
        experiment_status[job_id] = {
            "status": "running",
            "progress": 0,
            "total": 0,
            "completed": 0,
            "failed": 0,
            "current": "",
            "logs": []
        }
        
        config = load_config()
        if "error" in config:
            experiment_status[job_id]["status"] = "error"
            experiment_status[job_id]["logs"].append("Error loading config")
            return
        
        completed = 0
        failed = 0
        total = 0
        
        # Run Organoid experiments
        if analyzer_type in ["organoid", "all"]:
            all_experiments = []
            for experiment_config in config.get("organoid_experiments", []):
                combinations = generate_combinations(experiment_config, test_mode)
                for i, combination in enumerate(combinations):
                    experiment_id = f"organoid_{experiment_config['name']}_{i+1}"
                    output_dir = os.path.join(RESULTS_DIR, experiment_id)
                    all_experiments.append((experiment_id, combination, output_dir))
            
            experiments_to_run = []
            for experiment_id, combination, output_dir in all_experiments:
                if not check_experiment_exists(output_dir, "organoid"):
                    experiments_to_run.append((experiment_id, combination, output_dir))
            
            total += len(experiments_to_run)
            experiment_status[job_id]["total"] = total
            
            for idx, (experiment_id, combination, output_dir) in enumerate(experiments_to_run):
                experiment_status[job_id]["current"] = f"Running {experiment_id}"
                experiment_status[job_id]["progress"] = int((idx / len(experiments_to_run)) * 100) if experiments_to_run else 0
                
                cmd = [
                    "python", "-m", "organoid_analyzer.training.train_and_test_models",
                    "--seq_len", str(combination.get('seq_len', 100)),
                    "--epochs", str(combination.get('epochs', 400)),
                    "--batch_size", str(combination.get('batch_size', 256)),
                    "--dropout", str(combination.get('dropout', 0.3)),
                    "--hidden_sizes", str(combination.get('hidden_sizes', 32)),
                    "--fusion_sizes", str(combination.get('fusion_sizes', 64)),
                    "--model_type", str(combination.get('model_type', 'fusion')),
                    "--dataset", str(combination.get('dataset', 'all')),
                    "--output_dir", str(output_dir)
                ]
                
                # Run from project root so module imports work
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
                
                # Check if experiment actually produced results (more reliable than return code)
                experiment_succeeded = result.returncode == 0 or check_experiment_exists(output_dir, "organoid")
                
                if experiment_succeeded:
                    completed += 1
                    experiment_status[job_id]["logs"].append(f"✅ {experiment_id} completed")
                else:
                    failed += 1
                    error_msg = result.stderr[:200] if result.stderr else "Unknown error"
                    experiment_status[job_id]["logs"].append(f"❌ {experiment_id} failed: {error_msg}")
                
                experiment_status[job_id]["completed"] = completed
                experiment_status[job_id]["failed"] = failed
        
        # Run TIL experiments
        if analyzer_type in ["til", "all"]:
            all_experiments = []
            for experiment_config in config.get("til_experiments", []):
                combinations = generate_combinations(experiment_config, test_mode)
                for i, combination in enumerate(combinations):
                    experiment_id = f"til_{experiment_config['name']}_{i+1}"
                    output_dir = os.path.join(RESULTS_DIR, experiment_id)
                    all_experiments.append((experiment_id, combination, output_dir))
            
            experiments_to_run = []
            for experiment_id, combination, output_dir in all_experiments:
                if not check_experiment_exists(output_dir, "til"):
                    experiments_to_run.append((experiment_id, combination, output_dir))
            
            total += len(experiments_to_run)
            experiment_status[job_id]["total"] = total
            
            for idx, (experiment_id, combination, output_dir) in enumerate(experiments_to_run):
                experiment_status[job_id]["current"] = f"Running {experiment_id}"
                experiment_status[job_id]["progress"] = int(((completed + idx) / total) * 100) if total > 0 else 0
                
                cmd = [
                    "python", os.path.join(PROJECT_ROOT, "til_analyzer", "main_test.py"),
                    "--intermediate_features", str(combination.get('intermediate_features', 512)),
                    "--second_intermediate_features", str(combination.get('second_intermediate_features', 512)),
                    "--dropout_prob", str(combination.get('dropout_prob', 0.5)),
                    "--batch_size", str(combination.get('batch_size', 16)),
                    "--epochs", str(combination.get('epochs', 200)),
                    "--learning_rate", str(combination.get('learning_rate', 0.00025)),
                    "--model_type", str(combination.get('model_type', 'resnet18')),
                    "--dataset", str(combination.get('dataset', 'chip')),
                    "--output_dir", str(output_dir)
                ]
                
                # Run from project root
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
                
                # Check if experiment actually produced results (more reliable than return code 
                # since Python warnings go to stderr and can cause false failures)
                experiment_succeeded = result.returncode == 0 or check_experiment_exists(output_dir, "til")
                
                if experiment_succeeded:
                    completed += 1
                    experiment_status[job_id]["logs"].append(f"✅ {experiment_id} completed")
                else:
                    failed += 1
                    error_msg = result.stderr[:200] if result.stderr else "Unknown error"
                    experiment_status[job_id]["logs"].append(f"❌ {experiment_id} failed: {error_msg}")
                
                experiment_status[job_id]["completed"] = completed
                experiment_status[job_id]["failed"] = failed
        
        experiment_status[job_id]["status"] = "completed"
        experiment_status[job_id]["progress"] = 100
        
    except Exception as e:
        experiment_status[job_id]["status"] = "error"
        experiment_status[job_id]["logs"].append(f"Error: {str(e)}")

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    return jsonify(load_config())

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration."""
    try:
        config = request.json
        save_config(config)
        return jsonify({"success": True, "message": "Config updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/experiments/run', methods=['POST'])
def run_experiments():
    """Start running experiments."""
    data = request.json
    analyzer_type = data.get('analyzer', 'all')
    test_mode = data.get('test', False)
    
    job_id = str(uuid.uuid4())
    
    thread = threading.Thread(target=run_experiments_async, args=(analyzer_type, test_mode, job_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True, "job_id": job_id})

@app.route('/api/experiments/status/<job_id>', methods=['GET'])
def get_experiment_status(job_id):
    """Get experiment status."""
    if job_id in experiment_status:
        return jsonify(experiment_status[job_id])
    else:
        return jsonify({"error": "Job not found"}), 404

def calculate_survival_metrics(df):
    """Calculate meaningful survival analysis metrics from TIL data."""
    metrics = {}
    
    try:
        metrics['n_samples'] = len(df)
        metrics['n_timepoints'] = len(df.columns) - 1
        
        survival_data = df.iloc[:, 1:]
        survival_data_numeric = survival_data.apply(pd.to_numeric, errors='coerce')
        survival_data_numeric = survival_data_numeric.dropna(axis=1, how='all')
        
        if survival_data_numeric.empty:
            metrics['error'] = "No numeric data found in survival columns"
            return metrics
        
        mean_survival = survival_data_numeric.mean()
        n_timepoints = len(mean_survival)
        
        idx_1yr = min(7, n_timepoints - 1) if n_timepoints > 0 else 0
        idx_2yr = min(14, n_timepoints - 1) if n_timepoints > 0 else 0
        idx_final = n_timepoints - 1 if n_timepoints > 0 else 0
        
        metrics['mean_survival_1yr'] = float(mean_survival.iloc[idx_1yr]) if n_timepoints > 0 else None
        metrics['mean_survival_2yr'] = float(mean_survival.iloc[idx_2yr]) if n_timepoints > 0 else None
        metrics['mean_survival_final'] = float(mean_survival.iloc[idx_final]) if n_timepoints > 0 else None
        
        survival_var = survival_data_numeric.var()
        metrics['survival_variance'] = float(survival_var.mean()) if not survival_var.empty else 0
        
        diff_matrix = survival_data_numeric.diff(axis=1).abs()
        metrics['curve_smoothness'] = float(diff_matrix.mean().mean()) if not diff_matrix.empty else 0
        
        survival_std = survival_data_numeric.std()
        metrics['survival_discrimination'] = float(survival_std.mean()) if not survival_std.empty else 0
        
        metrics['model_confidence'] = 1.0 / (metrics['survival_variance'] + 1e-8)
        
        quality_score = (
            (1.0 / (metrics['curve_smoothness'] + 1e-8)) * 0.3 +
            metrics['survival_discrimination'] * 0.4 +
            metrics['model_confidence'] * 0.3
        )
        metrics['quality_score'] = quality_score
        
    except Exception as e:
        metrics['error'] = str(e)
    
    return metrics

def generate_summary_plots(results_dir):
    """Generate summary plots for all experiments."""
    organoid_results = []
    til_results = []
    
    if not os.path.exists(results_dir):
        return False
    
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        if item.startswith("organoid_"):
            accuracies_dir = os.path.join(item_path, "ablation_Specify", "accuracies")
            if os.path.exists(accuracies_dir):
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
                        test_acc = float(df.iloc[-1, 1]) if len(df) > 1 else 0
                        organoid_results.append({
                            'experiment': item,
                            'test_accuracy': test_acc
                        })
                    except:
                        pass
        
        elif item.startswith("til_"):
            metrics_dir = os.path.join(item_path, "metrics")
            if os.path.exists(metrics_dir):
                xlsx_files = [f for f in os.listdir(metrics_dir) if f.endswith('.xlsx')]
                if xlsx_files:
                    latest_xlsx = sorted(xlsx_files)[-1]
                    xlsx_path = os.path.join(metrics_dir, latest_xlsx)
                    try:
                        df = pd.read_excel(xlsx_path)
                        metrics = calculate_survival_metrics(df)
                        til_results.append({
                            'experiment': item,
                            'metrics': metrics
                        })
                    except:
                        pass
    
    # Generate plots
    if organoid_results or til_results:
        try:
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
            return True
            
        except Exception as e:
            print(f"Error generating plots: {e}")
            return False
    
    return False

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get experiment results summary."""
    results = {
        "organoid": [],
        "til": []
    }
    
    if not os.path.exists(RESULTS_DIR):
        return jsonify(results)
    
    for item in os.listdir(RESULTS_DIR):
        item_path = os.path.join(RESULTS_DIR, item)
        if not os.path.isdir(item_path):
            continue
        
        if item.startswith("organoid_"):
            accuracies_dir = os.path.join(item_path, "ablation_Specify", "accuracies")
            if os.path.exists(accuracies_dir):
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
                        test_acc = float(df.iloc[-1, 1]) if len(df) > 1 else 0
                        results["organoid"].append({
                            "name": item,
                            "test_accuracy": test_acc,
                            "timestamp": latest_time
                        })
                    except:
                        pass
        
        elif item.startswith("til_"):
            metrics_dir = os.path.join(item_path, "metrics")
            if os.path.exists(metrics_dir):
                xlsx_files = [f for f in os.listdir(metrics_dir) if f.endswith('.xlsx')]
                if xlsx_files:
                    latest_xlsx = sorted(xlsx_files)[-1]
                    xlsx_path = os.path.join(metrics_dir, latest_xlsx)
                    try:
                        df = pd.read_excel(xlsx_path)
                        metrics = calculate_survival_metrics(df)
                        results["til"].append({
                            "name": item,
                            "timestamp": os.path.getmtime(xlsx_path),
                            "metrics": metrics
                        })
                    except:
                        pass
    
    # Generate summary plots
    generate_summary_plots(RESULTS_DIR)
    
    return jsonify(results)

@app.route('/api/results/summary', methods=['GET'])
def get_results_summary():
    """Get results summary plot."""
    summary_path = os.path.join(RESULTS_DIR, 'experiment_summary.png')
    if os.path.exists(summary_path):
        return send_from_directory(os.path.dirname(summary_path), os.path.basename(summary_path), mimetype='image/png')
    else:
        # Try to generate it
        generate_summary_plots(RESULTS_DIR)
        if os.path.exists(summary_path):
            return send_from_directory(os.path.dirname(summary_path), os.path.basename(summary_path), mimetype='image/png')
        return jsonify({"error": "Summary not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

