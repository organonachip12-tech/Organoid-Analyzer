# Organoid Analyzer & TIL Analyzer

This repository contains two complementary biomedical analysis tools for cancer research:

- **Organoid Analyzer**: Analyzes cell tracking data from microscopy videos to predict treatment responses
- **TIL Analyzer**: Analyzes tumor-infiltrating lymphocyte (TIL) images for survival prediction

## Overview

### Organoid Analyzer
The Organoid Analyzer processes cell tracking data from microscopy videos to predict treatment responses. It uses a fusion model that combines:
- **Sequential features**: Time-series data from cell trajectories (area, perimeter, speed, etc.)
- **Track-level features**: Statistical summaries of entire cell tracks
- **Machine learning**: LSTM with attention mechanism for sequence processing

### TIL Analyzer  
The TIL Analyzer analyzes tumor-infiltrating lymphocyte images to predict patient survival outcomes. It uses:
- **Deep learning**: ResNet18-based neural network for image feature extraction
- **Survival analysis**: Neural network survival modeling for time-to-event prediction
- **Visualization**: UMAP projections and survival curve generation

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for both projects)
- Fiji/ImageJ (for Organoid Analyzer cell tracking)

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Organoid-Analyzer-1
```

2. **Create and activate virtual environment:**
```bash
# Create virtual environment
python -m venv organoid-til-env

# Activate environment
# On macOS/Linux:
source organoid-til-env/bin/activate
# On Windows:
# organoid-til-env\Scripts\activate
```

3. **Install dependencies:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

4. **Install Fiji/ImageJ (for Organoid Analyzer):**
- Download Fiji from https://fiji.sc/
- Update the `FIJI_PATH` in `Organoid Analyzer/Config.py`

## Data Formats

### Organoid Analyzer Input Data

#### Required Directory Structure:
```
Data/
├── CART annotations.xlsx          # CART treatment annotations
├── 2nd batch annotations.xlsx     # Second batch annotations  
├── PDO_annotation.xlsx            # PDO device annotations
├── CART/                         # CART treatment data
│   ├── NYU318/
│   ├── NYU352/
│   └── ...
├── 2ND/                          # Second batch data
└── PDO/                          # PDO device data
```

#### Annotation File Format (Excel):
- **CART**: Sheet "Summary" with columns: ID, Label (0-2 scale)
- **2ND**: Sheet 0 with columns: "Meso IL18 CAR T cells", "Labels" 
- **PDO**: Sheet "Statistics" with columns: "Name", "Score"

#### Cell Tracking Data Format:
Each subfolder should contain:
- **TrackMate files**: `.xml` files with cell tracking data
- **Spot files**: `.csv` files with spot measurements
- **Image sequences**: Microscopy video frames

#### Features Extracted:
- **Sequential features**: AREA, PERIMETER, CIRCULARITY, ELLIPSE_ASPECTRATIO, SOLIDITY, SPEED, MEAN_SQUARE_DISPLACEMENT, RADIUS
- **Track features**: TRACK_DISPLACEMENT, TRACK_STD_SPEED, MEAN_DIRECTIONAL_CHANGE_RATE

### TIL Analyzer Input Data

#### Required Directory Structure:
```
TIL-Analyzer-main/
├── images/                       # TIL images
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── clinical.csv                  # Clinical data
└── annotations.csv               # Generated annotations
```

#### Clinical Data Format (CSV):
```csv
Patient_ID,Image_Filename,Survival_Time,Death_Status,...
P001,image1.png,365,Dead,...
P002,image2.png,730,Alive,...
```

#### Generated Annotations Format (CSV):
```csv
image_path,survival_time,death_occurred
./images/image1.png,365,1
./images/image2.png,730,0
```

## Usage

### Organoid Analyzer

1. **Prepare your data:**
   - Place microscopy videos in appropriate folders
   - Create annotation Excel files with treatment labels
   - Update paths in `Config.py`

2. **Run cell tracking (if needed):**
```bash
cd "Organoid Analyzer"
python track_cells.py
```

3. **Create dataset:**
```bash
python create_dataset.py
```

4. **Train models:**
```bash
python train_and_test_models.py
```

5. **Test trained model:**
```bash
python test_model.py
```

### TIL Analyzer

1. **Prepare your data:**
   - Place TIL images in `images/` folder
   - Create `clinical.csv` with patient data
   - Update paths in the scripts

2. **Generate annotations:**
```bash
cd TIL-Analyzer-main
python generateAnnotations.py
```

3. **Train survival model:**
```bash
python main_test.py
```

4. **Generate visualizations:**
```bash
python generateImage.py
```

## Configuration

### Organoid Analyzer Configuration (`Config.py`)

Key settings to modify:
- `FIJI_PATH`: Path to Fiji installation
- `DATA_DIR`: Input data directory
- `CELL_TRACKING_DATASET_CONFIGS`: Dataset-specific settings
- `DATASET_CONFIGS`: Annotation file paths
- `SEQ_LEN`: Number of frames per sequence (default: 100)
- `MAX_EPOCHS`: Training epochs (default: 400)
- `BATCH_SIZE`: Training batch size (default: 256)

### TIL Analyzer Configuration

Key settings to modify in `main_test.py`:
- `high_surv_annotations`: Path to high survival annotations
- `med_surv_annotations`: Path to medium survival annotations  
- `low_surv_annotations`: Path to low survival annotations
- `train_annotations_path`: Path to training annotations
- Model parameters: `num_intervals`, `intermediate_features`, `dropout_prob`

## Output Files

### Organoid Analyzer Outputs:
- `Generated/models/`: Trained model files
- `Generated/trajectory_dataset_100.npz`: Processed sequence data
- `Generated/track_dataset.npz`: Processed track data
- `Results/`: Analysis results and plots
- `unscaled_spot_features.csv`: Raw feature data
- `unscaled_track_features.csv`: Raw track data

### TIL Analyzer Outputs:
- `validate_survivals/`: Survival analysis results
- `results/`: Model performance metrics
- `survival_curve.png`: Survival probability plots
- `UMAP/`: Dimensionality reduction visualizations
- `metrics*.xlsx`: Statistical analysis results

## Model Architectures

### Organoid Analyzer: UnifiedFusionModel
- **LSTM**: Bidirectional LSTM for sequence processing
- **Attention**: Attention mechanism for temporal focus
- **Fusion**: Combines sequential and track-level features
- **Classification**: 3-class output (treatment response levels)

### TIL Analyzer: ResNet18-based Survival Model
- **Backbone**: ResNet18 (pretrained) for feature extraction
- **Survival Head**: Neural network survival modeling
- **Output**: Survival probability over time intervals

## Troubleshooting

### Common Issues:

1. **CUDA out of memory:**
   - Reduce `BATCH_SIZE` in Config.py
   - Use smaller `SEQ_LEN` for sequences

2. **Fiji/ImageJ not found:**
   - Update `FIJI_PATH` in Config.py
   - Ensure Fiji is properly installed

3. **Missing dependencies:**
   - Install PyTorch with correct CUDA version
   - Check all requirements.txt packages

4. **Data format errors:**
   - Verify Excel file sheet names match Config.py
   - Check CSV column headers match expected format

### Performance Tips:

1. **GPU Usage:**
   - Both projects benefit significantly from GPU acceleration
   - Monitor GPU memory usage during training

2. **Data Preprocessing:**
   - Organoid Analyzer: Ensure cell tracking data is clean
   - TIL Analyzer: Verify image quality and annotations

3. **Model Training:**
   - Start with smaller datasets for testing
   - Monitor training curves for overfitting

## Citation

If you use these tools in your research, please cite the original papers and acknowledge this implementation.

## License

See LICENSE files in each project directory for licensing information.

## Unified Experiment System

The repository includes a unified experiment system that allows you to run multiple configurations of both analyzers in parallel.

### Quick Start with Experiments

1. **Run all experiments:**
```bash
python run_experiments.py --config experiments_config.json
```

2. **Run only Organoid experiments:**
```bash
python run_experiments.py --config experiments_config.json --analyzer organoid
```

3. **Run only TIL experiments:**
```bash
python run_experiments.py --config experiments_config.json --analyzer til
```

### Experiment Configuration

Edit `experiments_config.json` to define your experiments:

```json
{
  "global_settings": {
    "max_parallel_experiments": 2,
    "output_dir": "./experiments_results"
  },
  "organoid_experiments": [
    {
      "name": "organoid_baseline",
      "hidden_sizes": [16, 32, 64],
      "fusion_sizes": [32, 64, 128],
      "dropout": [0.3, 0.5],
      "seq_len": [50, 100],
      "batch_size": [128, 256],
      "epochs": [200, 400]
    }
  ],
  "til_experiments": [
    {
      "name": "til_baseline",
      "intermediate_features": [256, 512, 1024],
      "dropout_prob": [0.3, 0.5],
      "batch_size": [16, 32],
      "epochs": [200, 400],
      "learning_rate": [0.00025, 0.001]
    }
  ]
}
```

### Experiment Results

Results are organized in `experiments_results/`:
```
experiments_results/
├── organoid_experiments/
│   ├── organoid_baseline_20241201_143022_a1b2c3d4/
│   │   ├── config.json          # Experiment parameters
│   │   ├── stdout.log           # Training output
│   │   ├── stderr.log           # Error logs
│   │   └── Results/             # Model outputs
│   └── ...
└── til_experiments/
    ├── til_baseline_20241201_143100_i9j0k1l2/
    └── ...
```

### Benefits of the Experiment System

- **Parallel execution**: Run multiple experiments simultaneously
- **Easy configuration**: JSON-based experiment setup
- **Isolated results**: Each experiment has its own directory
- **Comprehensive logging**: All outputs and errors saved
- **Unified interface**: Single command for both analyzers
- **Resource management**: Limits parallel experiments to prevent overload

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the configuration files for proper setup
3. Ensure all dependencies are correctly installed
4. Verify data formats match the expected structure
5. Check the experiment logs in `experiments_results/` for debugging
