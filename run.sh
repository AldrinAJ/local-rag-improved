#!/bin/bash

echo "Starting AI Document Assistant..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found"
    exit 1
fi

# Check if requirements are installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Installing requirements..."
    if ! pip3 install -r requirements.txt; then
        echo "Error: Failed to install requirements"
        exit 1
    fi
fi

# Start the application
echo "Starting Streamlit application..."
python3 -m streamlit run Welcome.py

echo "Application stopped."