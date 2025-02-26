#!/bin/bash

# Define the main TeX file
MAIN_TEX="main.tex"

# Check if the main.tex file exists
if [ ! -f "$MAIN_TEX" ]; then
  echo "Error: $MAIN_TEX not found!"
  exit 1
fi

# Compile the TeX file using pdflatex
pdflatex "$MAIN_TEX"

# Check if the compilation was successful
if [ $? -eq 0 ]; then
  echo "Compilation successful!"
else
  echo "Compilation failed!"
fi
