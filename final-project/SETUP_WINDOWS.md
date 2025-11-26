# Windows Setup Guide

## Prerequisites

### 1. Install Anaconda or Miniconda

Download and install from:
- **Anaconda**: https://www.anaconda.com/products/distribution
- **Miniconda**: https://docs.conda.io/en/latest/miniconda.html

During installation, make sure to:
- ✅ Check "Add Anaconda to PATH" (or add manually later)
- ✅ Install for all users (if you have admin rights)

### 2. Verify Installation

Open **Anaconda Prompt** or **Command Prompt** and run:

```cmd
conda --version
python --version
```

If these commands work, you're ready to proceed.

## Setup Steps

### Step 1: Open Terminal

- **Option A**: Open **Anaconda Prompt** (recommended)
- **Option B**: Open **Command Prompt** or **PowerShell**

### Step 2: Set NOAA Token (required for NOAA data)

1. Request a NOAA CDO API token: <https://www.ncei.noaa.gov/cdo-web/token>
2. Set the token as an environment variable (replace the sample token with yours):
   ```cmd
   setx NOAA_TOKEN "ktMwlDFeStWCbALFlwQGbFMHTBYmzdpO"
   ```
3. Close and reopen the terminal so the variable is available.

### Step 3: Navigate to Project Directory

```cmd
cd C:\Users\86132\Desktop\final-project\final-project
```

### Step 4: Create Conda Environment

```cmd
conda env create -f environment.yml
```

This will create an environment named `climate` with all required packages.

### Step 5: Activate Environment

```cmd
conda activate climate
```

You should see `(climate)` in your prompt.

## Running the Project

### Method 1: Using Batch Script (Easiest)

Simply double-click `run.bat` or run:

```cmd
run.bat
```

This will automatically:
1. Activate the environment
2. Download real data from Berkeley Earth
3. Train the models
4. Evaluate the models

### Method 2: Manual Step-by-Step

```cmd
# Activate environment
conda activate climate

# Step 1: Download real data
python -m src.prepare_data --config configs/default.yaml --use-real-data

# Step 2: Train models
python -m src.train --config configs/default.yaml

# Step 3: Evaluate models
python -m src.evaluate --config configs/default.yaml
```

### Method 3: Using Make (if available)

If you have `make` installed (via Git Bash or WSL):

```bash
make data
make train
make evaluate
```

## Troubleshooting

### Issue: "conda is not recognized"

**Solution:**
1. Reinstall Anaconda/Miniconda with "Add to PATH" option
2. Or manually add to PATH:
   - Add: `C:\Users\YourUsername\anaconda3\Scripts`
   - Add: `C:\Users\YourUsername\anaconda3`
3. Restart terminal after adding to PATH

### Issue: "Python was not found"

**Solution:**
1. Make sure you activated the environment: `conda activate climate`
2. Verify Python in environment: `python --version`
3. If still not working, recreate environment:
   ```cmd
   conda env remove -n climate
   conda env create -f environment.yml
   ```

### Issue: "No module named 'src'"

**Solution:**
1. Make sure you're in the project root directory:
   ```cmd
   cd C:\Users\86132\Desktop\final-project\final-project
   ```
2. Check current directory: `cd`
3. Verify `src` folder exists: `dir src`

### Issue: Data download fails

**Solution:**
1. Check internet connection
2. Try manual download (see RUN_GUIDE.md)
3. The script will generate synthetic data as fallback

### Issue: "Permission denied" or "Access denied"

**Solution:**
1. Run terminal as Administrator
2. Or install to user directory instead of system directory

## Verifying Setup

After setup, verify everything works:

```cmd
conda activate climate
python -c "import pandas, numpy, torch, statsmodels; print('All packages installed!')"
```

If this prints "All packages installed!" without errors, you're ready to go!

## Next Steps

Once setup is complete, see [RUN_GUIDE.md](RUN_GUIDE.md) for detailed running instructions.


