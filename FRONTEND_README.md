# Experiment Manager Frontend

A clean web-based frontend for managing and running Organoid & TIL experiments.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install Flask flask-cors
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Start the Web Server

```bash
python app.py
```

The frontend will be available at: **http://localhost:5000**

## 📋 Features

### Configuration Tab
- View and edit experiment configuration
- Save changes to `experiments_config.json`
- Format JSON for readability
- Reload configuration from file

### Run Experiments Tab
- Select analyzer type (All, Organoid, or TIL)
- Enable test mode for quick testing
- Real-time progress tracking
- Live logs showing experiment status
- Progress bar and completion statistics

### Results Tab
- View all completed experiments
- See test accuracies for Organoid experiments
- View TIL experiment results
- Automatic refresh when experiments complete

## 🎨 Interface

The frontend features a modern, clean design with:
- Gradient header with clear branding
- Tab-based navigation
- Real-time status updates
- Color-coded logs (green for success, red for errors)
- Responsive design for different screen sizes

## 🔧 How It Works

1. **Configuration**: Edit your experiment parameters in the Configuration tab
2. **Run**: Start experiments from the Run Experiments tab
3. **Monitor**: Watch real-time progress and logs
4. **Results**: View completed experiment results in the Results tab

The backend runs experiments in background threads, so you can continue using the interface while experiments are running.

## 📝 Notes

- Experiments run in the background - you can close the browser and they'll continue
- The interface polls for status updates every second
- All functionality from `run_experiments.py` is available through the web interface
- Results are automatically displayed when experiments complete

