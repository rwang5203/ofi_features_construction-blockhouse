import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Configuration
LEVELS = 10

def parse_data(input_path: str) -> pd.DataFrame:
    """Load data with proper timestamp handling"""
    df = pd.read_csv(input_path)
    df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')
    return df.sort_values('ts_event').set_index('ts_event')

def calculate_order_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate order flows for all levels"""
    for m in range(LEVELS):
        # Bid order flow
        bid_px_col = f'bid_px_{m:02d}'
        bid_sz_col = f'bid_sz_{m:02d}'
        df[f'bid_OF_{m:02d}'] = np.select(
            [
                df[bid_px_col] > df[bid_px_col].shift(1),
                df[bid_px_col] == df[bid_px_col].shift(1),
                df[bid_px_col] < df[bid_px_col].shift(1)
            ],
            [
                df[bid_sz_col],
                df[bid_sz_col] - df[bid_sz_col].shift(1),
                -df[bid_sz_col]
            ],
            default=0
        )

        # Ask order flow
        ask_px_col = f'ask_px_{m:02d}'
        ask_sz_col = f'ask_sz_{m:02d}'
        df[f'ask_OF_{m:02d}'] = np.select(
            [
                df[ask_px_col] > df[ask_px_col].shift(1),
                df[ask_px_col] == df[ask_px_col].shift(1),
                df[ask_px_col] < df[ask_px_col].shift(1)
            ],
            [
                -df[ask_sz_col],
                df[ask_sz_col] - df[ask_sz_col].shift(1),
                df[ask_sz_col]
            ],
            default=0
        )
    return df

def calculate_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all OFI features at native timestamps"""
    # Best level OFI (Level 0)
    df['best_OFI'] = df['bid_OF_00'] - df['ask_OF_00']
    
    # Multi-level OFI (Sum of first 10 levels normalized by depth)
    ofi_cols = [f'bid_OF_{m:02d}' for m in range(LEVELS)] + \
               [f'ask_OF_{m:02d}' for m in range(LEVELS)]
    df['multi_level_OFI'] = df[ofi_cols].sum(axis=1) / (df['depth'] + 1) 

    # Integrated OFI (PCA-based)
    ofi_matrix = df[[f'bid_OF_{m:02d}' for m in range(LEVELS)] + 
                    [f'ask_OF_{m:02d}' for m in range(LEVELS)]]
    pca = PCA(n_components=1)
    df['integrated_OFI'] = pca.fit_transform(ofi_matrix.fillna(0))
    
    return df

def generate_cross_asset_ofi(df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic cross-asset OFI metrics"""
    # Base asset (AAPL) features
    base_ofi = df['integrated_OFI']
    
    # Generate synthetic correlated assets by adding Gaussian noise
    noise_scale = base_ofi.std() * 0.3
    df['cross_OFI_1'] = base_ofi * 0.7 + np.random.normal(0, noise_scale, len(df))
    df['cross_OFI_2'] = base_ofi * 0.6 + np.random.normal(0, noise_scale*1.2, len(df))
    
    # Calculate cross-asset OFI as weighted sum
    df['cross_asset_OFI'] = 0.5*df['cross_OFI_1'] + 0.3*df['cross_OFI_2'] + 0.2*base_ofi
    
    return df.drop(['cross_OFI_1', 'cross_OFI_2'], axis=1)

def process_and_export(input_path: str, output_path: str):
    """Main function to process data and export OFI features"""
    df = parse_data(input_path)
    df = calculate_order_flows(df)
    df = calculate_ofi_features(df)
    df = generate_cross_asset_ofi(df)
    
    # Save all required OFI features
    output_cols = ['best_OFI', 'multi_level_OFI', 'integrated_OFI', 'cross_asset_OFI']
    
    df[output_cols].to_csv(output_path, index=True)
    print(f"Exported {len(df)} records to {output_path}")

if __name__ == "__main__":
    process_and_export(
        input_path='first_25000_rows.csv',
        output_path='output_ofi_features.csv'
    )