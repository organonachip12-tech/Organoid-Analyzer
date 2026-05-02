# Quick Start Guide

## 🚀 Getting Started

### 1. Environment Setup

**Option A: Using venv (Recommended)**
```bash
# Create virtual environment
python -m venv organoid-til-env

# Activate environment
# On macOS/Linux:
source organoid-til-env/bin/activate
# On Windows:
# organoid-til-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (automatically detects your platform)
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements.txt

# Run setup script
python setup_venv.py
```

### 2. Data Preparation

#### For Organoid Analyzer:
1. **Create data structure:**
   ```
   Data/
   ├── CART annotations.xlsx
   ├── 2nd batch annotations.xlsx
   ├── PDO_annotation.xlsx
   ├── CART/
   │   ├── NYU318/
   │   ├── NYU352/
   │   └── ...
   ├── 2ND/
   └── PDO/
   ```

2. **Annotation format (Excel files):**
   - **CART**: Sheet "Summary" with ID and Label columns
   - **2ND**: Sheet 0 with "Meso IL18 CAR T cells" and "Labels" columns
   - **PDO**: Sheet "Statistics" with "Name" and "Score" columns

3. **Update Config.py:**
   ```python
   FIJI_PATH = r"/path/to/Fiji.app"  # Update this path
   ```

#### For TIL Analyzer:
1. **Create data structure:**
   ```
   TIL-Analyzer-main/
   ├── images/
   │   ├── image1.png
   │   ├── image2.png
   │   └── ...
   └── clinical.csv
   ```

2. **Clinical data format (CSV):**
   ```csv
   Patient_ID,Image_Filename,Survival_Time,Death_Status
   P001,image1.png,365,Dead
   P002,image2.png,730,Alive
   ```

### 3. Running Analysis

#### Organoid Analyzer:
```bash
cd "Organoid Analyzer"

# Step 1: Create dataset (if you have raw tracking data)
python create_dataset.py

# Step 2: Train models
python train_and_test_models.py

# Step 3: Test trained model
python test_model.py
```

#### TIL Analyzer:
```bash
cd TIL-Analyzer-main

# Step 1: Generate annotations
python generateAnnotations.py

# Step 2: Train survival model
python main_test.py

# Step 3: Generate visualizations
python generateImage.py
```

#### GigaTIME:
```bash
# Option 1: Web UI (easiest)
python run_frontend.py
# Open http://localhost:5000 → GigaTIME tab → upload tile or slide

# Option 2: CLI inference
python -m gigatime_analyzer.training.main --mode infer \
  --tiling_dir ./data/gigatime/preprocessed_tiles \
  --metadata ./data/gigatime/preprocessed_tiles/preprocessed_metadata.csv
```

The pretrained model downloads automatically from HuggingFace on first run. Place `model.pth` in `data/gigatime/` to skip download, or set `HF_TOKEN` for gated repos.

## 🔧 Troubleshooting

### Common Issues:

1. **CUDA out of memory:**
   - Reduce `BATCH_SIZE` in Config.py
   - Use smaller `SEQ_LEN` for sequences

2. **Fiji not found:**
   - Download from https://fiji.sc/
   - Update `FIJI_PATH` in Config.py

3. **Missing dependencies:**
   - From the repo root: `pip install -e .` (editable install; dependencies come from `requirements.txt`)
   - Optional full setup (dirs + verify): `organoid-dev-bootstrap` or `python -m organoid_analyzer.dev_bootstrap`
   - Check CUDA version compatibility

4. **Data format errors:**
   - Verify Excel sheet names match Config.py
   - Check CSV column headers

5. **GigaTIME model download fails:**
   - Set `HF_TOKEN` environment variable if the HuggingFace repo is gated
   - Or manually download `model.pth` and place in `data/gigatime/`

### Performance Tips:

- **GPU**: Both projects benefit significantly from GPU acceleration
- **Memory**: Monitor GPU memory usage during training
- **Data**: Ensure clean, properly formatted input data

## 📊 Expected Outputs

### Organoid Analyzer:
- `Generated/models/`: Trained model files
- `Generated/trajectory_dataset_100.npz`: Processed sequence data
- `Results/`: Analysis results and plots

### TIL Analyzer:
- `validate_survivals/`: Survival analysis results
- `results/`: Model performance metrics
- `survival_curve.png`: Survival probability plots
- `UMAP/`: Dimensionality reduction visualizations

### GigaTIME:
- Web UI: Channel maps, stats table, CSV/JSON export
- CLI: `results/gigatime/` with metrics and checkpoints

## 🆘 Need Help?

1. Check the full README.md for detailed documentation
2. Verify your data formats match the examples above
3. Ensure all dependencies are properly installed
4. Check the troubleshooting section for common issues
