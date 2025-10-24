# Unified Experiment System

## 🚀 Quick Start

### 1. Run All Experiments
```bash
python run_experiments.py --config experiments_config.json
```

### 2. Run Only Organoid Experiments
```bash
python run_experiments.py --config experiments_config.json --analyzer organoid
```

### 3. Run Only TIL Experiments
```bash
python run_experiments.py --config experiments_config.json --analyzer til
```

## 📁 File Structure Created
```
experiments_results/
├── organoid_experiments/
│   ├── organoid_baseline_20241201_143022_a1b2c3d4/
│   │   ├── config.json          # Experiment parameters
│   │   ├── stdout.log           # Training output
│   │   ├── stderr.log           # Error logs
│   │   └── Results/             # Model outputs
│   └── organoid_feature_ablation_20241201_143045_e5f6g7h8/
└── til_experiments/
    ├── til_baseline_20241201_143100_i9j0k1l2/
    └── til_architecture_test_20241201_143115_m3n4o5p6/
```

## ⚙️ Configuration

Edit `experiments_config.json` to modify experiments:

```json
{
  "global_settings": {
    "max_parallel_experiments": 2,
    "output_dir": "./experiments_results"
  },
  "organoid_experiments": [
    {
      "name": "my_experiment",
      "hidden_sizes": [16, 32],
      "fusion_sizes": [64, 128],
      "dropout": [0.3],
      "seq_len": [100],
      "batch_size": [256],
      "epochs": [200]
    }
  ]
}
```

## 🔧 What Was Modified

**Minimal changes made to existing code:**

1. **`Organoid Analyzer/train_and_test_models.py`**: Added command-line argument support
2. **`TIL-Analyzer-main/main_test.py`**: Added command-line argument support
3. **New files created**: `run_experiments.py`, `experiments_config.json`

**No existing functionality was changed** - the original scripts work exactly as before.

## 📊 Results

- Each experiment runs in its own directory
- All logs and outputs are saved
- Summary report generated automatically
- Easy to compare different configurations

## 🎯 Benefits

- **Parallel execution**: Run multiple experiments simultaneously
- **Easy configuration**: JSON-based experiment setup
- **Isolated results**: Each experiment has its own directory
- **Minimal code changes**: Existing functionality preserved
- **Unified interface**: Single command for both analyzers
