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

## 🔧 Troubleshooting

### Common Issues:

1. **CUDA out of memory:**
   - Reduce `BATCH_SIZE` in Config.py
   - Use smaller `SEQ_LEN` for sequences

2. **Fiji not found:**
   - Download from https://fiji.sc/
   - Update `FIJI_PATH` in Config.py

3. **Missing dependencies:**
   - Run `python setup.py` to verify installation
   - Check CUDA version compatibility

4. **Data format errors:**
   - Verify Excel sheet names match Config.py
   - Check CSV column headers

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

## 🆘 Need Help?

1. Check the full README.md for detailed documentation
2. Verify your data formats match the examples above
3. Ensure all dependencies are properly installed
4. Check the troubleshooting section for common issues
