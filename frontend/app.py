#!/usr/bin/env python3
"""
Flask Web Frontend for Experiment Management
"""

import json
import os
import subprocess
import sys
import threading
import time
import itertools
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory, Response, stream_with_context
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
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "organoid_analyzer", "training", "train_and_test_models.py"),
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
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
                
                # Check if experiment actually produced results (more reliable than return code)
                experiment_succeeded = result.returncode == 0 or check_experiment_exists(output_dir, "organoid")
                
                if experiment_succeeded:
                    completed += 1
                    experiment_status[job_id]["logs"].append(f"✅ {experiment_id} completed")
                else:
                    failed += 1
                    err = (result.stderr or "").strip()
                    out = (result.stdout or "").strip()
                    error_msg = err if err else "Unknown error"
                    if len(error_msg) > 2000:
                        error_msg = error_msg[:2000] + "\n... (truncated)"
                    if out:
                        error_msg = error_msg + "\n[stdout]\n" + (out[:500] if len(out) > 500 else out)
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
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "til_analyzer", "main_test.py"),
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
                    err = (result.stderr or "").strip()
                    out = (result.stdout or "").strip()
                    error_msg = err if err else "Unknown error"
                    if len(error_msg) > 2000:
                        error_msg = error_msg[:2000] + "\n... (truncated)"
                    if out:
                        error_msg = error_msg + "\n[stdout]\n" + (out[:500] if len(out) > 500 else out)
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

# ─── GigaTIME Tab ────────────────────────────────────────────────────────────

import base64
import io

_gigatime_model = None
_gigatime_device = None

GIGATIME_CHANNELS = [
    "DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34",
    "CD68", "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1",
    "CK", "Ki67", "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin"
]


def _load_gigatime_model():
    global _gigatime_model, _gigatime_device
    if _gigatime_model is not None:
        return _gigatime_model, _gigatime_device
    import torch
    from gigatime_analyzer.models.archs import gigatime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gigatime(num_classes=23, input_channels=3)
    # Compatibility shim for older PyTorch checkpoint serialization.
    # The checkpoint pickle may reference arbitrary nested attrs/calls on this
    # module, so we use a recursive stub that is both callable and attr-accessible.
    import types as _types

    class _Permissive:
        """Object that returns itself for any attribute access or call."""
        def __getattr__(self, name):
            return _Permissive()
        def __call__(self, *a, **kw):
            return _Permissive()

    class _PermissiveModule(_types.ModuleType):
        def __getattr__(self, name):
            return _Permissive()

    if "torch.utils.serialization" not in sys.modules:
        _ser_mod = _PermissiveModule("torch.utils.serialization")
        _ser_config = _PermissiveModule("torch.utils.serialization.config")
        _ser_mod.config = _ser_config
        sys.modules["torch.utils.serialization"] = _ser_mod
        sys.modules["torch.utils.serialization.config"] = _ser_config

    ckpt_path = os.path.join(PROJECT_ROOT, "data", "gigatime", "model.pth")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
    else:
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN")
        ckpt_path = hf_hub_download(
            repo_id="prov-gigatime/GigaTIME",
            filename="model.pth",
            token=hf_token,
            local_dir=os.path.join(PROJECT_ROOT, "data", "gigatime"),
        )
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    _gigatime_model = model
    _gigatime_device = device
    return model, device


@app.route('/api/gigatime/predict', methods=['POST'])
def gigatime_predict():
    """Run GigaTIME inference on an uploaded H&E tile."""
    import torch
    import numpy as np
    from torchvision import transforms
    from PIL import Image as PILImage

    if 'tile' not in request.files:
        return jsonify({"error": "No tile file provided"}), 400

    file = request.files['tile']
    try:
        img = PILImage.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {e}"}), 400

    try:
        model, device = _load_gigatime_model()
    except Exception as e:
        return jsonify({"error": f"Model load failed: {e}"}), 500

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        output = torch.sigmoid(output)

    output_np = output.squeeze(0).cpu().numpy()  # (23, 512, 512)

    input_resized = img.resize((512, 512))
    buf = io.BytesIO()
    input_resized.save(buf, format='PNG')
    input_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    channel_images = []
    for i, name in enumerate(GIGATIME_CHANNELS):
        ch = output_np[i]
        ch_uint8 = (ch * 255).clip(0, 255).astype(np.uint8)
        ch_img = PILImage.fromarray(ch_uint8, mode='L')
        buf = io.BytesIO()
        ch_img.save(buf, format='PNG')
        channel_images.append({
            "name": name,
            "image": base64.b64encode(buf.getvalue()).decode('utf-8'),
        })

    return jsonify({"input": input_b64, "channels": channel_images})


@app.route('/api/gigatime/predict/stream', methods=['POST'])
def gigatime_predict_stream():
    """SSE-streaming GigaTIME inference with per-step progress."""
    if 'tile' not in request.files:
        return jsonify({"error": "No tile file provided"}), 400
    file_bytes = request.files['tile'].read()

    def generate():
        import torch
        import numpy as np
        import json as _json
        from torchvision import transforms
        from PIL import Image as PILImage

        def event(step, total, message, result=None):
            payload = {"step": step, "total": total, "message": message}
            if result is not None:
                payload["result"] = result
            return f"data: {_json.dumps(payload)}\n\n"

        # Step 1: Validate image
        yield event(1, 6, "Validating image\u2026")
        try:
            img = PILImage.open(io.BytesIO(file_bytes)).convert('RGB')
        except Exception as e:
            yield event(0, 6, f"Error: cannot read image \u2014 {e}")
            return

        # Step 2: Load model (may download on first run)
        yield event(2, 6, "Loading model (downloading on first run)\u2026")
        try:
            model, device = _load_gigatime_model()
        except Exception as e:
            yield event(0, 6, f"Error: model load failed \u2014 {e}")
            return

        # Step 3: Preprocess
        yield event(3, 6, "Preprocessing image\u2026")
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(img).unsqueeze(0).to(device)

        # Step 4: Inference
        yield event(4, 6, "Running inference\u2026")
        with torch.no_grad():
            output = model(tensor)
            output = torch.sigmoid(output)
        output_np = output.squeeze(0).cpu().numpy()

        # Step 5: Encode channels (prob map, binary map, per-channel stats)
        yield event(5, 6, "Encoding 23 channels\u2026")
        input_resized = img.resize((512, 512))
        buf = io.BytesIO()
        input_resized.save(buf, format='PNG')
        input_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        channel_images = []
        for i, name in enumerate(GIGATIME_CHANNELS):
            ch = output_np[i]  # sigmoid probabilities, float32 [0,1]

            # probability map (grayscale visualisation)
            prob_img = PILImage.fromarray((ch * 255).clip(0, 255).astype(np.uint8), mode='L')
            buf = io.BytesIO()
            prob_img.save(buf, format='PNG')
            prob_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # binary thresholded map (threshold 0.5, same as evaluation)
            bin_img = PILImage.fromarray(((ch > 0.5).astype(np.uint8) * 255), mode='L')
            buf = io.BytesIO()
            bin_img.save(buf, format='PNG')
            bin_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            channel_images.append({
                "name":   name,
                "image":  prob_b64,
                "binary": bin_b64,
                "stats": {
                    "mean_prob":    round(float(ch.mean()), 4),
                    "max_prob":     round(float(ch.max()),  4),
                    "min_prob":     round(float(ch.min()),  4),
                    "std_prob":     round(float(ch.std()),  4),
                    "pct_positive": round(float((ch > 0.5).mean()) * 100, 2),
                },
            })

        # Step 6: Done — send full result
        yield event(6, 6, "Inference complete \u2014 23 channels generated.",
                    result={"input": input_b64, "channels": channel_images})

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


_SLIDE_EXTS = {'.svs', '.ndpi', '.tiff', '.tif', '.scn', '.mrxs', '.vms', '.vmu'}


@app.route('/api/gigatime/process_slide/stream', methods=['POST'])
def gigatime_process_slide_stream():
    """
    SSE endpoint: accept an uploaded whole-slide image, save it, preprocess
    it into tiles, run GigaTIME inference on every tile, stream progress.

    Request: multipart/form-data with field 'slide' (SVS/NDPI/TIFF/etc.)

    SSE event payload:
        { "phase", "message", "tile_n", "tile_total", "result" }
    """
    if 'slide' not in request.files:
        return jsonify({"error": "No slide file provided"}), 400

    file = request.files['slide']
    filename = file.filename or "slide"
    suffix = Path(filename).suffix.lower()
    if suffix not in _SLIDE_EXTS:
        return jsonify({"error": f"Unsupported format '{suffix}'. Use SVS, NDPI, TIFF, SCN, MRXS."}), 400

    # Save uploaded file to data/gigatime/svs/ before streaming begins
    svs_dir = Path(PROJECT_ROOT) / "data" / "gigatime" / "svs"
    svs_dir.mkdir(parents=True, exist_ok=True)
    svs_path = svs_dir / filename
    file.save(str(svs_path))

    def generate():
        import json as _json
        import torch
        import numpy as np
        from torchvision import transforms
        from PIL import Image as PILImage

        def evt(phase, message, tile_n=None, tile_total=None, result=None):
            payload = {"phase": phase, "message": message,
                       "tile_n": tile_n, "tile_total": tile_total,
                       "result": result}
            return f"data: {_json.dumps(payload)}\n\n"

        # ── Phase 1: file saved ───────────────────────────────────────────
        yield evt("preprocess", f"Slide saved ({svs_path.stat().st_size // (1024*1024)} MB): {svs_path.name}")

        # ── Phase 2: preprocess slide → tiles ────────────────────────────
        yield evt("preprocess", "Preprocessing slide into tiles (this may take several minutes)\u2026")
        tiles_dir = Path(PROJECT_ROOT) / "data" / "gigatime" / "preprocessed_tiles"
        try:
            from gigatime_analyzer.preprocessing import process_slide
            num_tiles, slide_out_dir = process_slide(svs_path, tiles_dir)
        except ImportError as _ie:
            yield evt("error",
                "OpenSlide is not available. "
                "Run: pip install openslide-bin  "
                "(this installs the Windows DLLs automatically). "
                f"Detail: {_ie}")
            return
        except Exception as e:
            yield evt("error", f"Preprocessing failed: {e}")
            return

        if num_tiles == 0:
            yield evt("error", "No tiles passed quality filters. Check tissue threshold or slide quality.")
            return

        yield evt("preprocess", f"Preprocessing complete — {num_tiles} tiles extracted.", tile_n=0, tile_total=num_tiles)

        # ── Phase 3: load model ───────────────────────────────────────────
        yield evt("infer", "Loading GigaTIME model\u2026", tile_n=0, tile_total=num_tiles)
        try:
            model, device = _load_gigatime_model()
        except Exception as e:
            yield evt("error", f"Model load failed: {e}")
            return

        # ── Phase 4: infer on each tile, accumulate per-channel stats ─────
        tile_paths = sorted(slide_out_dir.glob("*_he.png"))
        total = len(tile_paths)

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # accumulators: shape (23,) running mean/variance (Welford)
        n_processed = 0
        ch_mean_acc  = np.zeros(len(GIGATIME_CHANNELS), dtype=np.float64)
        ch_m2_acc    = np.zeros(len(GIGATIME_CHANNELS), dtype=np.float64)
        ch_max_acc   = np.full(len(GIGATIME_CHANNELS), -np.inf, dtype=np.float64)
        ch_pct_acc   = np.zeros(len(GIGATIME_CHANNELS), dtype=np.float64)  # sum of pct_positive

        # spatial accumulator: list of (row, col, mean_per_channel vec)
        spatial_records = []

        for idx, tile_path in enumerate(tile_paths):
            yield evt("infer", f"Inferring tile {idx + 1}/{total}: {tile_path.name}",
                      tile_n=idx + 1, tile_total=total)
            try:
                img = PILImage.open(tile_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = torch.sigmoid(model(tensor))
                ch_np = out.squeeze(0).cpu().numpy()  # (23, 512, 512)

                # Welford online mean/variance per channel
                n_processed += 1
                tile_ch_means = np.zeros(len(GIGATIME_CHANNELS), dtype=np.float32)
                for c in range(len(GIGATIME_CHANNELS)):
                    tile_mean = float(ch_np[c].mean())
                    tile_ch_means[c] = tile_mean
                    delta = tile_mean - ch_mean_acc[c]
                    ch_mean_acc[c] += delta / n_processed
                    ch_m2_acc[c]   += delta * (tile_mean - ch_mean_acc[c])
                    if ch_np[c].max() > ch_max_acc[c]:
                        ch_max_acc[c] = float(ch_np[c].max())
                    ch_pct_acc[c] += float((ch_np[c] > 0.5).mean()) * 100

                # Parse spatial position from filename: {x_l0}_{y_l0}_{size}_{size}_he.png
                parts = tile_path.stem.split('_')  # stem removes _he.png suffix → x_y_s_s
                # stem is like "12345_67890_556_556_he" (since extension is .png, stem = everything before .png)
                # Actually tile_path.stem = "12345_67890_556_556_he"
                # parts: ['12345', '67890', '556', '556', 'he']
                if len(parts) >= 2:
                    try:
                        x_l0 = int(parts[0])
                        y_l0 = int(parts[1])
                        spatial_records.append((x_l0, y_l0, tile_ch_means))
                    except ValueError:
                        pass

            except Exception:
                pass  # skip corrupt tiles silently

        if n_processed == 0:
            yield evt("error", "All tiles failed during inference.")
            return

        # ── Phase 5: build spatial heatmaps ───────────────────────────────
        yield evt("infer", "Building spatial probability maps…",
                  tile_n=n_processed, tile_total=total)

        spatial_maps = []
        if spatial_records:
            xs = [r[0] for r in spatial_records]
            ys = [r[1] for r in spatial_records]
            # Determine tile step size (use most common gap or min non-zero)
            unique_xs = sorted(set(xs))
            unique_ys = sorted(set(ys))
            step_x = min((b - a for a, b in zip(unique_xs, unique_xs[1:])), default=556)
            step_y = min((b - a for a, b in zip(unique_ys, unique_ys[1:])), default=556)
            if step_x <= 0: step_x = 556
            if step_y <= 0: step_y = 556

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            n_cols = int(round((x_max - x_min) / step_x)) + 1
            n_rows = int(round((y_max - y_min) / step_y)) + 1

            # Build grid: shape (n_rows, n_cols, 23), NaN where no tile
            grid = np.full((n_rows, n_cols, len(GIGATIME_CHANNELS)), np.nan, dtype=np.float32)
            for x_l0, y_l0, ch_means in spatial_records:
                col = int(round((x_l0 - x_min) / step_x))
                row = int(round((y_l0 - y_min) / step_y))
                if 0 <= row < n_rows and 0 <= col < n_cols:
                    grid[row, col, :] = ch_means

            # Render each channel as a viridis heatmap
            import io as _io
            cmap = plt.cm.viridis
            for c, name in enumerate(GIGATIME_CHANNELS):
                ch_grid = grid[:, :, c]  # (n_rows, n_cols)
                fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
                # Mask NaN tiles (no tissue)
                masked = np.ma.array(ch_grid, mask=np.isnan(ch_grid))
                im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect='auto',
                               interpolation='nearest')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                buf = _io.BytesIO()
                fig.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.05)
                plt.close(fig)
                buf.seek(0)
                spatial_maps.append({
                    "name": name,
                    "image": base64.b64encode(buf.getvalue()).decode('utf-8'),
                })

        # ── Phase 6: aggregate stats and return ───────────────────────────
        channel_stats = []
        for c, name in enumerate(GIGATIME_CHANNELS):
            variance = ch_m2_acc[c] / n_processed if n_processed > 1 else 0.0
            channel_stats.append({
                "name":         name,
                "mean_prob":    round(float(ch_mean_acc[c]), 4),
                "std_prob":     round(float(variance ** 0.5), 4),
                "max_prob":     round(float(ch_max_acc[c]), 4),
                "pct_positive": round(float(ch_pct_acc[c] / n_processed), 2),
            })

        # Sort by mean_prob descending so most-expressed channels appear first
        channel_stats.sort(key=lambda x: x["mean_prob"], reverse=True)

        yield evt("done",
                  f"Complete \u2014 {n_processed} tiles processed.",
                  tile_n=n_processed, tile_total=total,
                  result={
                      "tiles_processed": n_processed,
                      "slide_name": svs_path.name,
                      "output_dir": str(slide_out_dir),
                      "channels": channel_stats,
                      "spatial_maps": spatial_maps,
                  })

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

