# OFI Feature Construction

This repository implements my implementation of OFI feature construction from limit order book (LOB) data.

## Features

- **Best-Level OFI**: Top-of-book imbalance
- **Multi-Level OFI**: Depth-scaled aggregate of 10 levels
- **Integrated OFI**: PCA-weighted combination of levels
- **Cross-Asset OFI**: Realistic synthetic market simulation

## Prerequisites

```bash
pip install pandas numpy scikit-learn
```

## Usage
```bash
python ofi_process.py
``` 

## Data Requirements
Input CSV must contain the following columns for at least 10 order book levels:
- `ts_event`: Nanosecond timestamp of market events
- `bid_px_00` to `bid_px_09`: Bid prices for levels 0-9
- `bid_sz_00` to `bid_sz_09`: Bid sizes for levels 0-9  
- `ask_px_00` to `ask_px_09`: Ask prices for levels 0-9
- `ask_sz_00` to `ask_sz_09`: Ask sizes for levels 0-9

## OFI Calculation Methods

### Best-Level OFI
Calculates order flow imbalance at the top of the book (level 0):
```python
best_OFI = bid_OF_00 - ask_OF_00
```
where bid/ask_OF_00 represents the net order flow at the best bid/ask price level.

### Multi-Level OFI
Computes depth-scaled order flow across multiple levels (10 in this case):
1. Calculate raw OFI for each level:
    ```python
    OFI_m = bid_OF_m - ask_OF_m
    ```
2. Scale and sum OFIs:
    ```python
    multi_level_OFI = sum(OFI_m / depth for m in 0-9)
    ```

### Integrated OFI
Generates PCA-weighted combination of multi-level OFI:
1. Perform PCA on scaled multi-level OFI matrix
2. Extract first principal component
3. Normalize weights:
    ```python
    weights = PCA_components / sum(abs(PCA_components))
    ```

### Cross-Asset OFI Simulation
Generates synthetic correlated assets `SYNTH1` and `SYNTH2` using:
```python
SYNTH_OFI = alpha * AAPL_OFI + epsilon
```
where
- alpha = correlation coefficient (0.7 for SYNTH1, 0.6 for SYNTH2)
- epsilon = gaussian noise ~ N(0, sigma^2) with sigma = 0.3 * std(AAPL_OFI)

Final cross-asset OFI is computed as:
```python
cross_asset_OFI = 0.5*SYNTH1 + 0.3*SYNTH2 + 0.2*AAPL_OFI
```

