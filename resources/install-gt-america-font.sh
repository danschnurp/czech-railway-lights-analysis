#!/bin/bash

# Script to install GT America font for fasthesis.cls on macOS
# For use with TeXStudio and Homebrew-installed TeX Live

# Exit on error
set -e

echo "GT America Font Installation for LaTeX (fasthesis.cls)"
echo "======================================================"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed. Please install it first:"
    echo "  /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Check if TeX Live is installed via Homebrew
if ! brew list --formula | grep -q "texlive"; then
    echo "TeX Live not found. Installing via Homebrew..."
    brew install texlive
fi

# Find the location of TEXMFLOCAL
TEXMFLOCAL=$(kpsewhich -var-value=TEXMFLOCAL)
if [ -z "$TEXMFLOCAL" ]; then
    echo "Could not determine TEXMFLOCAL location. Using default location..."
    TEXMFLOCAL="/usr/local/texlive/texmf-local"
fi

echo "Using TEXMFLOCAL directory: $TEXMFLOCAL"

# Create destination directories if they don't exist
mkdir -p "$TEXMFLOCAL/fonts/truetype/gtamerica"
mkdir -p "$TEXMFLOCAL/fonts/map/dvips/gtamerica"
mkdir -p "$TEXMFLOCAL/fonts/enc/dvips/gtamerica"
mkdir -p "$TEXMFLOCAL/fonts/tfm/gtamerica"
mkdir -p "$TEXMFLOCAL/fonts/vf/gtamerica"
mkdir -p "$TEXMFLOCAL/fonts/type1/gtamerica"
mkdir -p "$TEXMFLOCAL/tex/latex/gtamerica"

# Prompt user for the location of the fasthesis.zip file
echo ""
echo "Please enter the path to the fasthesis.zip file:"
read FASTHESIS_ZIP

if [ ! -f "$FASTHESIS_ZIP" ]; then
    echo "Error: File not found. Please verify the path."
    exit 1
fi

# Create a temporary directory for extraction
TEMP_DIR=$(mktemp -d)
echo "Extracting to temporary directory: $TEMP_DIR"

# Extract the zip file
unzip -q "$FASTHESIS_ZIP" -d "$TEMP_DIR"

# Check if the install directory exists in the extracted contents
if [ ! -d "$TEMP_DIR/install/\$TEXMFLOCAL" ]; then
    echo "Error: Expected directory structure not found in zip file."
    echo "Looking for: $TEMP_DIR/install/\$TEXMFLOCAL"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Copy the font files to the TEX installation
echo "Copying font files to $TEXMFLOCAL..."
cp -R "$TEMP_DIR/install/\$TEXMFLOCAL/fonts" "$TEXMFLOCAL/"
cp -R "$TEMP_DIR/install/\$TEXMFLOCAL/tex" "$TEXMFLOCAL/"

# Add the map file to updmap.cfg
echo "Updating font map configuration..."
if ! grep -q "Map GTAmerica" "$(kpsewhich updmap.cfg)"; then
    echo "Map GTAmerica" >> "$(kpsewhich updmap.cfg)"
    echo "Added 'Map GTAmerica' to updmap.cfg"
else
    echo "'Map GTAmerica' already exists in updmap.cfg"
fi

# Update the font maps
echo "Regenerating font maps..."
sudo mktexlsr
sudo updmap-sys --verbose

# Clean up
rm -rf "$TEMP_DIR"

echo ""
echo "GT America font has been installed successfully!"
echo ""
echo "TeXStudio Configuration:"
echo "1. Open TeXStudio"
echo "2. Go to Options > Configure TeXStudio"
echo "3. Select 'Commands' tab"
echo "4. Make sure the paths point to your Homebrew TeX installation"
echo "   Typically at: /usr/local/texlive/[year]/bin/[platform]"
echo "5. Click OK and restart TeXStudio"
echo ""
echo "You should now be able to use the fasthesis.cls with GT America font."
