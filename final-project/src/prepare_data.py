"""
Data preparation script
For downloading/generating and processing Berkeley Earth temperature anomaly data
"""
import argparse
import os
import yaml
from pathlib import Path
from src.data import (
    download_berkeley_earth_from_url,
    process_berkeley_data,
    download_noaa_data,
    process_noaa_data
)

def main():
    parser = argparse.ArgumentParser(description='Prepare Berkeley Earth temperature anomaly data')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download/generate data')
    parser.add_argument('--raw-only', action='store_true',
                       help='Only generate raw data, skip processing')
    parser.add_argument('--use-real-data', action='store_true', default=True,
                       help='Attempt to download real data from Berkeley Earth')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    
    # Define paths for Berkeley Earth
    raw_path = 'data/raw/berkeley_earth_global_temperature.csv'
    processed_path = cfg['data']['be_path']
    
    print('=' * 60)
    print('Berkeley Earth Data Preparation')
    print('=' * 60)
    
    # Step 1: Generate/download raw data
    print('\n[Step 1/2] Preparing raw data...')
    download_berkeley_earth_from_url(raw_path, force_download=args.force, use_real_data=args.use_real_data)
    
    if not args.raw_only:
        # Step 2: Process data
        print('\n[Step 2/2] Processing Berkeley Earth data...')
        df = process_berkeley_data(
            raw_path=raw_path,
            output_path=processed_path,
            date_col=cfg['data']['date_col'],
            target_col=cfg['data']['target_col']
        )
        
        print('\n' + '=' * 60)
        print('Berkeley Earth Data preparation complete!')
        print('=' * 60)
        print(f'Raw data: {raw_path}')
        print(f'Processed data: {processed_path}')
        print(f'  - Records: {len(df)}')
        print(f'  - Time range: {df["Date"].min()} to {df["Date"].max()}')
        print(f'  - Temperature anomaly range: {df["TempAnomaly"].min():.2f}°C to {df["TempAnomaly"].max():.2f}°C')
        print(f'  - Mean: {df["TempAnomaly"].mean():.2f}°C')
        print(f'  - Std: {df["TempAnomaly"].std():.2f}°C')
    else:
        print('\n[DONE] Only generated raw Berkeley Earth data (per --raw-only)')
    
    # Optional NOAA data preparation
    noaa_cfg = cfg.get('noaa', {})
    if noaa_cfg.get('enabled'):
        print('\n' + '=' * 60)
        print('NOAA GSOM Data Preparation')
        print('=' * 60)
        noaa_raw = noaa_cfg.get('raw_path', 'data/raw/noaa_gsom.csv')
        noaa_processed = noaa_cfg.get('processed_path', 'data/processed/noaa_gsom_monthly.csv')
        token = noaa_cfg.get('token')
        token_env_var = noaa_cfg.get('token_env_var', 'NOAA_TOKEN')
        if not token and token_env_var:
            token = os.getenv(token_env_var)
            if token:
                print(f'[INFO] NOAA token loaded from environment variable: {token_env_var}')
        if not token:
            print('[WARN] NOAA token not provided. Set it via environment variable or config for real data.')
            print('       Falling back to synthetic NOAA data if download fails.')
        
        download_noaa_data(
            output_path=noaa_raw,
            dataset=noaa_cfg.get('dataset', 'GSOM'),
            start_date=noaa_cfg.get('start_date', '1980-01-01'),
            end_date=noaa_cfg.get('end_date'),
            token=token,
            token_env_var=token_env_var,
            datatype=noaa_cfg.get('datatype', 'TAVG'),
            location=noaa_cfg.get('locationid'),
            station=noaa_cfg.get('stationid'),
            force_download=args.force
        )
        
        if not args.raw_only:
            noaa_df = process_noaa_data(noaa_raw, noaa_processed)
            print(f'[INFO] NOAA processed records: {len(noaa_df)}')
        else:
            print('[INFO] Skipping NOAA processing due to --raw-only flag')
    else:
        print('\n[INFO] NOAA data preparation disabled in config.')

if __name__ == '__main__':
    main()
