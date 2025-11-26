import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
from tqdm import tqdm
import os
import requests
from io import StringIO
import time

def download_berkeley_earth_from_url(output_path: str, force_download: bool = False, 
                                     use_real_data: bool = True):
    """
    Download global temperature anomaly data directly from Berkeley Earth official URL
    
    Args:
        output_path: Output file path
        force_download: Whether to force re-download
        use_real_data: Whether to attempt downloading real data (fallback to synthetic if fails)
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    if p.exists() and not force_download:
        print(f'[DATA] Data file already exists: {output_path}')
        return p
    
    # Berkeley Earth data URLs (global land + ocean temperature anomaly)
    # Updated with verified working URL from official Berkeley Earth S3 bucket
    berkeley_urls = [
        # Primary: Official Berkeley Earth S3 bucket (verified working URL)
        "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_complete.txt",
        # Backup: Berkeley Earth GitHub repository
        "https://raw.githubusercontent.com/BerkeleyEarth/Data/master/Global/Land_and_Ocean_complete.txt",
        # Alternative GitHub path
        "https://github.com/BerkeleyEarth/Data/raw/master/Global/Land_and_Ocean_complete.txt",
        # Official Berkeley Earth website
        "http://berkeleyearth.lbl.gov/auto/Global/Complete_TAVG_complete.txt",
    ]
    
    if use_real_data:
        print('[DOWNLOAD] Attempting to download real data from Berkeley Earth...')
        for idx, url in enumerate(berkeley_urls, 1):
            try:
                print(f'[DOWNLOAD] [{idx}/{len(berkeley_urls)}] Trying URL: {url}')
                response = requests.get(url, timeout=60, stream=True, 
                                       headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                
                # Check if we got actual data
                content = response.text
                if len(content) < 1000:
                    print(f'[WARN] Response too short ({len(content)} chars), skipping...')
                    continue
                
                # Berkeley Earth data format: lines starting with % are comments
                # Data columns are usually: year, month, anomaly, uncertainty, ...
                lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('%') and not line.startswith('#'):
                        # Skip empty lines and header-like lines
                        if not line.lower().startswith(('year', 'date', 'month')):
                            lines.append(line)
                
                print(f'[INFO] Found {len(lines)} data lines')
                
                if len(lines) > 100:  # Ensure sufficient data
                    # Parse data - try multiple approaches
                    data_str = '\n'.join(lines)
                    df = None
                    
                    # Berkeley Earth data format: Year, Month, Monthly Anomaly, Uncertainty, ...
                    # We need columns: Year, Month, and Monthly Anomaly (3rd column)
                    # Method 1: Try whitespace separator (most common for Berkeley Earth)
                    try:
                        # Read with whitespace separator, use first 3 columns
                        df = pd.read_csv(StringIO(data_str), sep='\s+', header=None, 
                                        engine='python', on_bad_lines='skip')
                        # Berkeley Earth format: Year, Month, Monthly Anomaly, Uncertainty, Annual, ...
                        # We only need first 3 columns
                        if len(df.columns) >= 3:
                            df = df.iloc[:, :3]  # Take first 3 columns
                            df.columns = ['Year', 'Month', 'Anomaly']
                            # Filter out NaN values in Year and Month
                            df = df.dropna(subset=['Year', 'Month'])
                            # Convert to numeric
                            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                            df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
                            df['Anomaly'] = pd.to_numeric(df['Anomaly'], errors='coerce')
                            df = df.dropna(subset=['Year', 'Month', 'Anomaly'])
                            print(f'[INFO] Parsed with whitespace separator: {len(df)} records')
                        else:
                            raise ValueError(f'Insufficient columns: {len(df.columns)}')
                    except Exception as e1:
                        print(f'[DEBUG] Whitespace separator failed: {str(e1)[:50]}')
                        df = None
                        
                        # Method 2: Try tab separator
                        try:
                            df = pd.read_csv(StringIO(data_str), sep='\t', header=None,
                                            engine='python', on_bad_lines='skip')
                            if len(df.columns) >= 3:
                                df = df.iloc[:, :3]
                                df.columns = ['Year', 'Month', 'Anomaly']
                                df = df.dropna(subset=['Year', 'Month'])
                                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                                df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
                                df['Anomaly'] = pd.to_numeric(df['Anomaly'], errors='coerce')
                                df = df.dropna(subset=['Year', 'Month', 'Anomaly'])
                                print(f'[INFO] Parsed with tab separator: {len(df)} records')
                            else:
                                raise ValueError(f'Insufficient columns: {len(df.columns)}')
                        except Exception as e2:
                            print(f'[DEBUG] Tab separator failed: {str(e2)[:50]}')
                            df = None
                    
                    if df is not None and len(df) > 100:
                        # Validate data
                        if 'Year' in df.columns and 'Month' in df.columns and 'Anomaly' in df.columns:
                            # Filter valid years (1850-2100) and months (1-12)
                            df = df[(df['Year'] >= 1850) & (df['Year'] <= 2100)]
                            df = df[(df['Month'] >= 1) & (df['Month'] <= 12)]
                            
                            # Create date column
                            df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1), 
                                                       errors='coerce')
                            df = df.dropna(subset=['Date', 'Anomaly'])
                            
                            # Select and rename columns
                            df = df[['Date', 'Anomaly']].rename(columns={'Anomaly': 'TempAnomaly'})
                            df = df.sort_values('Date').reset_index(drop=True)
                            
                            if len(df) > 100:
                                # Save data
                                df.to_csv(p, index=False)
                                print(f'[SUCCESS] Successfully downloaded Berkeley Earth data: {len(df)} records')
                                print(f'[INFO] Time range: {df["Date"].min()} to {df["Date"].max()}')
                                print(f'[INFO] Temperature anomaly range: {df["TempAnomaly"].min():.2f}°C to {df["TempAnomaly"].max():.2f}°C')
                                return p
                            else:
                                print(f'[WARN] After cleaning, only {len(df)} records remain (need >100)')
                        else:
                            print(f'[WARN] Missing required columns. Available: {list(df.columns)}')
                    else:
                        if df is None:
                            print(f'[WARN] Failed to parse data')
                        else:
                            print(f'[WARN] Insufficient records after parsing: {len(df)} (need >100)')
                else:
                    print(f'[WARN] Insufficient data lines: {len(lines)} (need >100)')
                    
            except requests.exceptions.RequestException as e:
                print(f'[WARN] Network error for URL {idx}: {str(e)[:100]}')
                continue
            except Exception as e:
                print(f'[WARN] Error processing URL {idx}: {str(e)[:100]}')
                continue
        
        print('[WARN] All Berkeley Earth URLs failed, generating synthetic data...')
    
    # If download fails, generate synthetic data
    return download_berkeley_earth_data(output_path, force_download)

def download_berkeley_earth_data(output_path: str, force_download: bool = False):
    """
    Generate Berkeley Earth-style synthetic data (used when real data cannot be downloaded)
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    if p.exists() and not force_download:
        print(f'[DATA] Data file already exists: {output_path}')
        return p
    
    print('[DATA] Generating Berkeley Earth-style synthetic data...')
    
    # Generate realistic Berkeley Earth-style data
    # Time range: 1850-2024 (approx 174 years, 2088 months)
    start_date = '1850-01-01'
    end_date = '2024-12-01'
    idx = pd.date_range(start_date, end_date, freq='MS')
    
    rng = np.random.default_rng(42)
    n = len(idx)
    
    # Long-term trend: simulate global warming trend (approx -0.5°C in 1850, +1.2°C in 2024)
    years = np.arange(n) / 12.0
    trend = -0.5 + (1.2 + 0.5) * (years / (years[-1]))
    # Add some nonlinear acceleration
    trend += 0.3 * (years / years[-1]) ** 2
    
    # Seasonal variation (12-month cycle)
    month_of_year = np.array([d.month for d in idx])
    season = 0.15 * np.sin(2 * np.pi * (month_of_year - 1) / 12)
    
    # Interannual variation (El Niño/La Niña, etc., ~3-7 year cycles)
    interannual = 0.1 * np.sin(2 * np.pi * years / 5.0) + \
                  0.05 * np.sin(2 * np.pi * years / 3.5)
    
    # Random noise (varies over time, earlier data has more uncertainty)
    noise_std = 0.15 * (1 - 0.5 * years / years[-1]) + 0.05
    noise = rng.normal(0, noise_std, n)
    
    # Combine all components
    temp_anomaly = trend + season + interannual + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': idx,
        'TempAnomaly': temp_anomaly
    })
    
    # Save to raw directory
    df.to_csv(p, index=False)
    print(f'[SAVE] Data saved to: {output_path}')
    print(f'[INFO] Data range: {df["Date"].min()} to {df["Date"].max()}, {len(df)} months')
    print(f'[INFO] Temperature anomaly range: {df["TempAnomaly"].min():.2f}°C to {df["TempAnomaly"].max():.2f}°C')
    
    return p

def load_raw_data(raw_path: str):
    """Load raw data from raw directory"""
    p = Path(raw_path)
    if not p.exists():
        raise FileNotFoundError(f'Raw data file does not exist: {raw_path}')
    
    df = pd.read_csv(p, parse_dates=['Date'])
    print(f'[LOAD] Loaded {len(df)} records from {raw_path}')
    return df

def process_berkeley_data(raw_path: str, output_path: str, 
                          date_col: str = 'Date', 
                          target_col: str = 'TempAnomaly'):
    """
    Process Berkeley Earth raw data
    
    Args:
        raw_path: Raw data path
        output_path: Processed data output path
        date_col: Date column name
        target_col: Target column name (temperature anomaly)
    """
    # Load raw data
    if Path(raw_path).exists():
        df = load_raw_data(raw_path)
    else:
        # If raw data doesn't exist, generate demo data
        print(f'[WARN] Raw data file does not exist: {raw_path}, will generate synthetic data')
        raw_dir = Path(raw_path).parent
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_berkeley_earth_data(raw_path)
        df = load_raw_data(raw_path)
    
    # Data cleaning
    df = df.sort_values(date_col).copy()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Remove missing values
    initial_len = len(df)
    df = df.dropna(subset=[date_col, target_col])
    if len(df) < initial_len:
        print(f'[CLEAN] Removed {initial_len - len(df)} records with missing values')
    
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Save processed data
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    df[[date_col, target_col]].to_csv(output_p, index=False)
    print(f'[SAVE] Processed data saved to: {output_path}')
    print(f'[INFO] Processed data: {len(df)} records, time range: {df[date_col].min()} to {df[date_col].max()}')
    
    return df

def download_noaa_data(output_path: str, dataset: str = 'GSOM', 
                       start_date: str = '2020-01-01', 
                       end_date: str = None,
                       token: str = None,
                       token_env_var: str = 'NOAA_TOKEN',
                       datatype: str = 'TAVG',
                       location: str = None,
                       station: str = None,
                       force_download: bool = False):
    """
    Download data from NOAA Climate Data Online (CDO)
    
    Args:
        output_path: Output file path
        dataset: Dataset type ('GSOM'=Global Summary of Month, 'GSOY'=Global Summary of Year, 'DAILY'=Daily Summaries)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), if None uses current date
        token: NOAA CDO API token (optional but recommended to increase download limits)
        force_download: Whether to force re-download
    
    Note: NOAA CDO API requires registration to get token, visit https://www.ncei.noaa.gov/cdo-web/token
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    if p.exists() and not force_download:
        print(f'[DATA] NOAA data file already exists: {output_path}')
        return p
    
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    print(f'[DOWNLOAD] Downloading data from NOAA (dataset: {dataset})...')
    print('[INFO] Note: NOAA CDO API requires token, visit https://www.ncei.noaa.gov/cdo-web/token to register')
    
    # NOAA CDO API endpoint
    base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    
    # Dataset ID mapping
    dataset_ids = {
        'GSOM': 'GLOBAL_SUMMARY_OF_MONTH',
        'GSOY': 'GLOBAL_SUMMARY_OF_YEAR',
        'DAILY': 'GHCND'  # Global Historical Climatology Network Daily
    }
    
    dataset_id = dataset_ids.get(dataset, 'GLOBAL_SUMMARY_OF_MONTH')
    
    # Build request parameters
    params = {
        'datasetid': dataset_id,
        'startdate': start_date,
        'enddate': end_date,
        'datatypeid': (datatype or 'TAVG').upper(),
        'limit': 1000,  # Maximum records per request
        'units': 'metric'
    }
    if location:
        params['locationid'] = location
    if station:
        params['stationid'] = station
    
    headers = {}
    resolved_token = token
    if not resolved_token and token_env_var:
        resolved_token = os.getenv(token_env_var)
        if resolved_token:
            print(f'[INFO] Using NOAA token from environment variable {token_env_var}')
    if resolved_token:
        headers['token'] = resolved_token
    else:
        print('[WARN] NOAA API token not provided. Set the token or environment variable before running.')
        print('       Visit https://www.ncei.noaa.gov/cdo-web/token to request a token.')
    
    try:
        all_data = []
        offset = 1
        
        while True:
            params['offset'] = offset
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    all_data.extend(data['results'])
                    print(f'[DOWNLOAD] Retrieved {len(all_data)} records...')
                    
                    # Check if more data available
                    if len(data['results']) < params['limit']:
                        break
                    offset += params['limit']
                    time.sleep(0.5)  # Avoid requesting too fast
                else:
                    break
            elif response.status_code == 401:
                print('[ERROR] NOAA API authentication failed, please check if token is correct')
                print('[INFO] Will generate synthetic NOAA data...')
                return generate_noaa_demo_data(output_path, start_date, end_date)
            else:
                print(f'[WARN] NOAA API request failed (status code: {response.status_code})')
                print('[INFO] Will generate synthetic NOAA data...')
                return generate_noaa_demo_data(output_path, start_date, end_date)
        
        if len(all_data) > 0:
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            # Process date and temperature data
            if 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'])
            if 'value' in df.columns:
                df['TempAnomaly'] = df['value'] / 100.0  # NOAA data is usually in hundredths of degrees
            
            df = df[['Date', 'TempAnomaly']].dropna().sort_values('Date').reset_index(drop=True)
            df.to_csv(p, index=False)
            print(f'[SUCCESS] Successfully downloaded NOAA data: {len(df)} records')
            return p
        else:
            print('[WARN] No NOAA data retrieved, generating synthetic data...')
            return generate_noaa_demo_data(output_path, start_date, end_date)
            
    except Exception as e:
        print(f'[ERROR] NOAA data download failed: {str(e)}')
        print('[INFO] Will generate synthetic NOAA data...')
        return generate_noaa_demo_data(output_path, start_date, end_date)

def generate_noaa_demo_data(output_path: str, start_date: str, end_date: str):
    """Generate NOAA-style synthetic data"""
    print('[DATA] Generating NOAA-style synthetic data...')
    idx = pd.date_range(start_date, end_date, freq='D')
    rng = np.random.default_rng(42)
    
    # Generate synthetic daily average temperature data
    trend = np.linspace(15, 16, len(idx))  # Base temperature trend
    season = 10 * np.sin(2 * np.pi * np.arange(len(idx)) / 365.25)  # Seasonality
    noise = rng.normal(0, 3, len(idx))  # Daily variation
    
    temp = trend + season + noise
    
    df = pd.DataFrame({
        'Date': idx,
        'TempAnomaly': temp - temp.mean()  # Convert to anomaly
    })
    
    p = Path(output_path)
    df.to_csv(p, index=False)
    print(f'[SAVE] Synthetic NOAA data saved: {output_path}')
    return p

def process_noaa_data(raw_path: str, output_path: str):
    """Process NOAA GSOM data to ensure clean Date/TempAnomaly columns"""
    p = Path(raw_path)
    if not p.exists():
        raise FileNotFoundError(f'NOAA raw data not found: {raw_path}. Run download_noaa_data first.')
    
    df = pd.read_csv(p, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['TempAnomaly'] = pd.to_numeric(df.get('TempAnomaly'), errors='coerce')
    df = df.dropna(subset=['Date', 'TempAnomaly'])
    
    if len(df) == 0:
        raise ValueError('NOAA processed data is empty after cleaning.')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df[['Date', 'TempAnomaly']].to_csv(output_path, index=False)
    print(f'[SAVE] Processed NOAA data saved to: {output_path}')
    print(f'[INFO] NOAA data samples: {len(df)}, range: {df["Date"].min()} to {df["Date"].max()}')
    return df

def load_or_make_demo_csv(path: str, use_real_data: bool = True):
    """
    Load or generate demo data (backward compatibility function)
    
    Args:
        path: Processed data path
        use_real_data: Whether to attempt downloading real Berkeley Earth data
    
    If processed data doesn't exist, will generate from raw data. If raw data doesn't exist, will attempt to download real data.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    # If processed data exists, load directly
    if p.exists():
        print(f'[LOAD] Loading processed data: {path}')
        return pd.read_csv(p, parse_dates=['Date'])
    
    # Otherwise, try to generate from raw data
    raw_path = Path('data/raw/berkeley_earth_global_temperature.csv')
    if not raw_path.exists():
        # Try to download real data
        print('[INFO] Raw data not found, attempting to download from Berkeley Earth...')
        download_berkeley_earth_from_url(str(raw_path), use_real_data=use_real_data)
    
    # Process data
    return process_berkeley_data(str(raw_path), path)

def moving_average(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """Calculate moving average"""
    df = df.sort_values('Date').copy()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df['MA'] = df[col].rolling(window=window, center=True, min_periods=window).mean()
    return df.dropna(subset=['MA'])
