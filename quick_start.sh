#!/bin/bash

# Aviation Question Generation System - Quick Start Script
# This script runs the complete pipeline from data preparation to model training

echo "========================================================"
echo "Aviation Question Generation System - Quick Start"
echo "========================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Install requirements
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt --break-system-packages 2>/dev/null || pip install -q -r requirements.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null

echo ""
echo "========================================================"
echo "Step 1: Data Preparation"
echo "========================================================"
python3 1_data_preparation.py

echo ""
echo "========================================================"
echo "Step 2: Model Training"
echo "========================================================"
echo "⚠️  This may take 30-45 minutes on CPU, 5-10 minutes on GPU"
echo ""
read -p "Continue with training? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 2_model_training.py
else
    echo "Skipping training. You can run it later with: python3 2_model_training.py"
fi

echo ""
echo "========================================================"
echo "Step 3: Evaluation"
echo "========================================================"
if [ -d "./aviation_qg_model/checkpoint-epoch-3" ]; then
    python3 3_evaluation.py
else
    echo "⚠️  No trained model found. Skipping evaluation."
    echo "Please train the model first with: python3 2_model_training.py"
fi

echo ""
echo "========================================================"
echo "✅ Setup Complete!"
echo "========================================================"
echo ""
echo "Next steps:"
echo "  1. Review the generated files:"
echo "     - aviation_train.json (training data)"
echo "     - aviation_val.json (validation data)"
echo "     - evaluation_results.json (if evaluation ran)"
echo ""
echo "  2. Start the web interface:"
echo "     python3 4_api_server.py"
echo ""
echo "  3. Open your browser to: http://127.0.0.1:5000"
echo ""
echo "========================================================"
