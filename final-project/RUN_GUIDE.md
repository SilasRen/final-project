# Running Guide - Using Real Data

This guide explains how to run the project and download real data from Berkeley Earth and NOAA.

## Prerequisites

1. **Install Conda** (if not already installed)
2. **Create the environment**:
```bash
conda env create -f environment.yml
conda activate climate
```

## Step-by-Step Execution

### Before You Start: Set NOAA Token (required for NOAA data)

NOAA's Climate Data Online API requires a personal token.

1. Request a token: <https://www.ncei.noaa.gov/cdo-web/token>
2. Once you have the token (e.g. `ktMwlDFeStWCbALFlwQGbFMHTBYmzdpO`), set it as an environment variable:

**Windows (Command Prompt):**
```cmd
setx NOAA_TOKEN "ktMwlDFeStWCbALFlwQGbFMHTBYmzdpO"
```
Close and reopen the terminal after running `setx`.

**PowerShell (current session only):**
```powershell
$env:NOAA_TOKEN="ktMwlDFeStWCbALFlwQGbFMHTBYmzdpO"
```

**macOS/Linux:**
```bash
export NOAA_TOKEN="ktMwlDFeStWCbALFlwQGbFMHTBYmzdpO"
```

You can also place the token directly in `configs/default.yaml` under `noaa.token`, but storing tokens in environment variables is safer.

### Step 1: Download Real Data

The project will automatically attempt to download real data from Berkeley Earth. Run:

```bash
# Method 1: Using Makefile
make data

# Method 2: Direct Python command
python -m src.prepare_data --config configs/default.yaml --use-real-data

# Method 3: Force re-download (if you want to update data)
python -m src.prepare_data --config configs/default.yaml --use-real-data --force
```

**What happens:**
- The script will try to download from Berkeley Earth's official data sources
- If download succeeds, you'll see: `[SUCCESS] Successfully downloaded Berkeley Earth data`
- Data will be saved to `data/raw/berkeley_earth_global_temperature.csv`
- Processed data will be saved to `data/processed/berkeley_1x1_monthly.csv`

**If download fails:**
- Check your internet connection
- The script will show error messages
- You can manually download data (see Manual Download section below)

### Step 2: Train Models

After data is prepared, train the models:

```bash
# Using Makefile
make train

# Or directly
python -m src.train --config configs/default.yaml
```

**What happens:**
- Loads processed data
- Trains SARIMAX and/or LSTM models based on configuration
- Saves trained models to `outputs/models/`
- Generates forecast plot to `outputs/figures/forecast_demo.png`

### Step 3: Evaluate Models

Evaluate the trained models:

```bash
# Using Makefile
make evaluate

# Or directly
python -m src.evaluate --config configs/default.yaml
```

**What happens:**
- Loads trained models from `outputs/models/`
- Evaluates on test data
- Generates evaluation plots:
  - `evaluation_forecast.png` - Time series forecast
  - `evaluation_scatter.png` - Predicted vs Actual
  - `evaluation_residuals.png` - Residual analysis
- Saves metrics to `evaluation_metrics.txt`

### Complete Pipeline

Run everything in one command:

```bash
make all
```

This executes: data preparation → training → evaluation

## Manual Data Download (If Automatic Download Fails)

If automatic download fails, you can manually download data:

### Option 1: Berkeley Earth Website

1. Visit: https://berkeleyearth.org/data/
2. Navigate to "Global Time Series Data"
3. Download "Land and Ocean" complete dataset
4. Save the file to: `data/raw/berkeley_earth_global_temperature.csv`
5. Ensure the file has columns: Year, Month, Anomaly (or similar)

### Option 2: Berkeley Earth GitHub

1. Visit: https://github.com/BerkeleyEarth/Data
2. Navigate to `Global/Land_and_Ocean_complete.txt`
3. Download the raw file
4. Save to: `data/raw/berkeley_earth_global_temperature.csv`

### Option 3: Direct URL Download

You can also download directly using wget or curl:

```bash
# Using wget (Linux/Mac)
wget -O data/raw/berkeley_earth_global_temperature.csv \
  https://raw.githubusercontent.com/BerkeleyEarth/Data/master/Global/Land_and_Ocean_complete.txt

# Using curl (Linux/Mac)
curl -o data/raw/berkeley_earth_global_temperature.csv \
  https://raw.githubusercontent.com/BerkeleyEarth/Data/master/Global/Land_and_Ocean_complete.txt

# Using PowerShell (Windows)
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/BerkeleyEarth/Data/master/Global/Land_and_Ocean_complete.txt" \
  -OutFile "data/raw/berkeley_earth_global_temperature.csv"
```

After manual download, run:
```bash
python -m src.prepare_data --config configs/default.yaml --raw-only
```

## Data Format Requirements

The raw data file should have the following format:
- Columns: Year, Month, Anomaly (or similar)
- Or: Date, TempAnomaly
- Data should be space-separated or tab-separated
- Comments (lines starting with %) will be automatically skipped

## Troubleshooting

### Issue: "Download failed" or "Connection timeout"

**Solutions:**
1. Check internet connection
2. Try manual download (see above)
3. The script will automatically generate synthetic data as fallback

### Issue: "No module named 'src'"

**Solution:**
```bash
# Make sure you're in the project root directory
cd final-project
# And the environment is activated
conda activate climate
```

### Issue: "Model file not found" during evaluation

**Solution:**
```bash
# Make sure you've run training first
make train
# Then run evaluation
make evaluate
```

### Issue: Data format errors

**Solution:**
- Check that the raw data file exists in `data/raw/`
- Verify the file format matches Berkeley Earth format
- You can inspect the file manually to check format

## Verifying Real Data

To verify you're using real data (not synthetic):

1. Check the data file:
```bash
# View first few lines
head data/raw/berkeley_earth_global_temperature.csv

# Check file size (real data should be > 100KB)
ls -lh data/raw/berkeley_earth_global_temperature.csv
```

2. Check the date range in the output:
- Real data should span from ~1850 to ~2024
- Synthetic data spans from 1850 to 2024 but with different patterns

3. Look for download success message:
```
[SUCCESS] Successfully downloaded Berkeley Earth data: XXXX records
```

## Configuration

You can modify `configs/default.yaml` to:
- Change model type: `sarimax`, `lstm`, or `sarimax_lstm`
- Adjust model hyperparameters
- Change data paths
- Modify training settings

## Expected Outputs

After running the complete pipeline, you should have:

```
data/
├── raw/
│   └── berkeley_earth_global_temperature.csv  (real data)
└── processed/
    └── berkeley_1x1_monthly.csv

outputs/
├── models/
│   ├── sarimax_model.pkl
│   └── lstm_model.pth
└── figures/
    ├── forecast_demo.png
    ├── evaluation_forecast.png
    ├── evaluation_scatter.png
    ├── evaluation_residuals.png
    └── evaluation_metrics.txt
```

## Next Steps

- Experiment with different model configurations
- Try different hyperparameters
- Extend to NOAA data for extreme events
- Add more evaluation metrics
- Visualize results in different ways


