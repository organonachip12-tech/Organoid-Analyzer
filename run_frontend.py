#!/usr/bin/env python3
"""
Convenience script to run the frontend from the project root.
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Change to project root directory
os.chdir(PROJECT_ROOT)

# Import and run the app
from frontend.app import app

if __name__ == '__main__':
    print("Starting Experiment Manager...")
    print(f"Project root: {PROJECT_ROOT}")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)

