"""
Preprocess databento mbp-10 data into FI-2010 format.

Input:  ../databento_1/xnas-itch-20210701-20210715.mbp-10.dbn.zst
Output: ../newdata/Train_Dst_NoAuction_DecPre_CF_7.txt
        ../newdata/Test_Dst_NoAuction_DecPre_CF_7.txt
        ../newdata/Test_Dst_NoAuction_DecPre_CF_8.txt
        ../newdata/Test_Dst_NoAuction_DecPre_CF_9.txt

FI-2010 format: 45 rows × N columns (space-separated)
  Rows 0-39:   LOB features (10 levels × 4 fields: ask_px, ask_sz, bid_px, bid_sz)
  Rows 40-44:  Labels for horizons k=1,2,3,5,10 (values 1=down, 2=stationary, 3=up)
"""
import numpy as np
import databento as db
from pathlib import Path
import os

# ── Config ──────────────────────────────────────────────────────
SAMPLING_STEP = 10           # sample every N mbp-10 events (FI-2010 uses ~10)
ALPHA = 0.002                # price movement threshold for labeling
HORIZONS = [1, 2, 3, 5, 10]  # prediction horizons

# Date split (10 trading days: Jul 1,2,6,7,8,9,12,13,14,15)
TRAIN_VAL_DAYS = [1, 2, 6, 7, 8, 9, 12, 13]  # first 8 days → 80/20 train/val split later
TEST_DAYS = [14, 15]                            # last 2 days → test

# Test file split: 3 groups of 2 stocks each (mimics FI-2010's CF_7/8/9)
TEST_STOCK_GROUPS = [
    ["SOFI", "NFLX"],    # → CF_7
    ["CSCO", "WING"],    # → CF_8
    ["SHLS", "LSTR"],    # → CF_9
]

OUTPUT_DIR = Path("../newdata")
DBN_PATH = Path("../databento_1/xnas-itch-20210701-20210715.mbp-10.dbn.zst")

# ── Load data ───────────────────────────────────────────────────
print("Loading DBN data...")
store = db.DBNStore.from_file(str(DBN_PATH.resolve()))
df = store.to_df()
print(f"Loaded {len(df):,} events, {df['symbol'].nunique()} symbols")

# ── Sample every N events ───────────────────────────────────────
print(f"Sampling every {SAMPLING_STEP} events...")
df_sampled = df.iloc[::SAMPLING_STEP].copy()
print(f"After sampling: {len(df_sampled):,} events")

# ── Build feature matrix (40 rows × T columns) ─────────────────
# Order per level: ask_px, ask_sz, bid_px, bid_sz (10 levels = 40 features)

def build_features(df_sampled):
    """Extract 40 LOB features per sample, return as (40, T) array."""
    feature_rows = []
    for level in range(10):
        feature_rows.extend([
            df_sampled[f"ask_px_{level:02d}"].values,
            df_sampled[f"ask_sz_{level:02d}"].values,
            df_sampled[f"bid_px_{level:02d}"].values,
            df_sampled[f"bid_sz_{level:02d}"].values,
        ])
    return np.array(feature_rows, dtype=np.float64)  # (40, T)

print("Building feature matrix...")
features = build_features(df_sampled)  # (40, T)
T = features.shape[1]
print(f"Features shape: {features.shape}")

# ── Build labels ────────────────────────────────────────────────
# mid_price = (best_bid + best_ask) / 2
best_bid = df_sampled["bid_px_00"].values
best_ask = df_sampled["ask_px_00"].values
mid_prices = (best_bid + best_ask) / 2.0

def compute_labels(mid_prices, horizons, alpha):
    """
    For each time t, compute labels for each horizon h in horizons.
    Label: 1=down, 2=stationary, 3=up (matching FI-2010).
    """
    T = len(mid_prices)
    labels = np.zeros((len(horizons), T), dtype=np.float64)
    labels[:] = np.nan  # fill tail with NaN initially

    for t in range(T):
        for i, h in enumerate(horizons):
            end_idx = t + h + 1
            if end_idx <= T:
                future_avg = np.mean(mid_prices[t+1:end_idx])
                pct_change = (future_avg - mid_prices[t]) / mid_prices[t]

                if pct_change > alpha:
                    labels[i, t] = 3.0   # up
                elif pct_change < -alpha:
                    labels[i, t] = 1.0   # down
                else:
                    labels[i, t] = 2.0   # stationary

    return labels

print("Computing labels...")
labels = compute_labels(mid_prices, HORIZONS, ALPHA)  # (5, T)
print(f"Labels shape: {labels.shape}")

# ── Combine features + labels ───────────────────────────────────
data = np.vstack([features, labels])  # (45, T)
print(f"Combined data shape: {data.shape}")

# Remove NaN tail (samples where labels couldn't be computed)
valid_mask = ~np.isnan(data).any(axis=0)
data = data[:, valid_mask]
print(f"After removing NaN: {data.shape}")

# Also filter df_sampled to match
df_sampled = df_sampled.iloc[valid_mask]
print(f"Filtered events: {len(df_sampled):,}")

# ── Per-stock Z-Score normalization ─────────────────────────────
# Different stocks have vastly different price scales (e.g., NFLX ~$500 vs SOFI ~$15).
# Normalize each stock independently using its own train+val mean/std.
print("Normalizing (per-stock Z-Score)...")
# Split dates
train_val_mask = df_sampled['ts_event'].dt.day.isin(TRAIN_VAL_DAYS).values
test_mask = df_sampled['ts_event'].dt.day.isin(TEST_DAYS).values

print(f"Train+Val samples: {train_val_mask.sum():,}")
print(f"Test samples:       {test_mask.sum():,}")

# Normalize per stock
all_symbols = sorted(df_sampled['symbol'].unique())
for sym in all_symbols:
    sym_mask = (df_sampled['symbol'] == sym).values
    sym_train_val = sym_mask & train_val_mask

    if sym_train_val.sum() == 0:
        continue

    # Feature rows 0-39 for this stock's train+val
    sym_features_tv = data[:40, sym_train_val]

    means = np.mean(sym_features_tv, axis=1, keepdims=True)
    stds = np.std(sym_features_tv, axis=1, keepdims=True)
    stds[stds == 0] = 1.0

    # Apply to all samples of this stock
    data[:40, sym_mask] = (data[:40, sym_mask] - means) / stds

feature_data = data[:40, :]
print(f"Normalized feature range: [{feature_data.min():.3f}, {feature_data.max():.3f}]")

# ── Split and save ──────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Train file (train+val days, code does 80/20 split) ---
train_data = data[:, train_val_mask]
train_path = OUTPUT_DIR / "Train_Dst_NoAuction_DecPre_CF_7.txt"
print(f"\nSaving training file ({train_data.shape[1]} columns)...")
np.savetxt(str(train_path.resolve()), train_data, fmt='%.7e')
print(f"  → {train_path} ({os.path.getsize(train_path)/1024/1024:.1f} MB)")

# --- Test files (3 groups by stock) ---
for group_idx, stock_list in enumerate(TEST_STOCK_GROUPS):
    stock_mask = df_sampled['symbol'].isin(stock_list).values
    group_mask = stock_mask & test_mask
    test_data = data[:, group_mask]

    cf_num = group_idx + 7  # CF_7, CF_8, CF_9
    test_path = OUTPUT_DIR / f"Test_Dst_NoAuction_DecPre_CF_{cf_num}.txt"
    print(f"Saving test file CF_{cf_num} ({test_data.shape[1]} columns, stocks={stock_list})...")
    np.savetxt(str(test_path.resolve()), test_data, fmt='%.7e')
    print(f"  → {test_path} ({os.path.getsize(test_path)/1024/1024:.1f} MB)")

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Done! Output files in", OUTPUT_DIR.resolve())
print(f"\nTo use in models, set UNZIPPED_DATA_DIR to:")
print(f"  '{OUTPUT_DIR.resolve()}/'")
print(f"\nLabel distribution (train+val):")
for i, h in enumerate(HORIZONS):
    row = train_data[40 + i, :]
    down = np.sum(row == 1)
    stat = np.sum(row == 2)
    up = np.sum(row == 3)
    total = len(row)
    print(f"  Horizon k={h:2d}: down={down/total:.1%}, stationary={stat/total:.1%}, up={up/total:.1%}")
