"""
fix_models.py
-------------
Fixes Keras version mismatch WITHOUT retraining.

The .keras file has all the weights — they're just stored under a different
internal path than the current Keras version expects. This script:
  1. Reads weights directly from the .keras zip
  2. Builds a fresh model with the current Keras version
  3. Stuffs the old weights in
  4. Saves a new .keras file that loads cleanly

Usage:
    python fix_models.py           # fix all coins
    python fix_models.py --coins BTC ETH   # fix subset
"""

import argparse
import zipfile
import json
import os
import numpy as np

MODELS_DIR = "models"
ALL_COINS  = ["BTC", "ETH", "BNB", "SOL", "XRP",
              "ADA", "DOGE", "AVAX", "MATIC", "DOT"]


# ── Build fresh model (same architecture as notebook) ─────────────────────────
def build_model(lookback: int = 60):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ── Extract all weight arrays from old .keras zip ────────────────────────────
def extract_weights_from_keras(path: str) -> list:
    """
    .keras files are zip archives. Weights are stored as .npy files
    inside a 'model.weights.h5' or flat npy layout depending on Keras version.
    This handles both layouts.
    """
    weights = {}
    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()
        print(f"    Archive contents: {names}")

        # Layout 1: flat .npy files (older Keras)
        npy_files = [n for n in names if n.endswith(".npy")]
        if npy_files:
            print(f"    Found {len(npy_files)} .npy weight files")
            for npy in npy_files:
                with z.open(npy) as f:
                    arr = np.load(f, allow_pickle=False)
                    weights[npy] = arr
            return list(weights.values())

        # Layout 2: model.weights.h5 (newer Keras)
        h5_files = [n for n in names if n.endswith(".h5") or n.endswith(".weights.h5")]
        if h5_files:
            import tempfile, h5py
            print(f"    Found h5 weight file: {h5_files[0]}")
            with z.open(h5_files[0]) as f:
                data = f.read()
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            arrays = []
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    arrays.append(np.array(obj))
            with h5py.File(tmp_path, "r") as hf:
                hf.visititems(collect)
            os.unlink(tmp_path)
            print(f"    Extracted {len(arrays)} weight tensors from h5")
            return arrays

    return []


# ── Smart weight assignment ───────────────────────────────────────────────────
def assign_weights(model, raw_weights: list) -> bool:
    """
    Try to match extracted weight arrays to model layers by shape.
    Returns True if successful.
    """
    expected = model.get_weights()
    print(f"    Model expects {len(expected)} weight tensors")
    print(f"    Got {len(raw_weights)} tensors from file")

    # Filter to only arrays whose shape matches something expected
    shape_map = {w.shape: w for w in raw_weights}
    matched = []
    for exp in expected:
        if exp.shape in shape_map:
            matched.append(shape_map[exp.shape])
        else:
            print(f"    ⚠️  No match for shape {exp.shape}")
            return False

    model.set_weights(matched)
    return True


def assign_weights_ordered(model, raw_weights: list) -> bool:
    """
    Fallback: assign by position if count matches exactly.
    """
    expected = model.get_weights()
    # Filter out scalar/empty arrays from h5 dumps
    filtered = [w for w in raw_weights if w.ndim > 0 and w.size > 0]

    if len(filtered) != len(expected):
        print(f"    Count mismatch: need {len(expected)}, got {len(filtered)}")
        # Try trimming to expected count
        if len(filtered) > len(expected):
            filtered = filtered[:len(expected)]
        else:
            return False

    # Check shapes match
    for i, (exp, got) in enumerate(zip(expected, filtered)):
        if exp.shape != got.shape:
            print(f"    Shape mismatch at index {i}: "
                  f"need {exp.shape}, got {got.shape}")
            return False

    model.set_weights(filtered)
    return True


# ── Fix one coin ──────────────────────────────────────────────────────────────
def fix_coin(coin: str) -> bool:
    src = f"{MODELS_DIR}/{coin}.keras"
    dst = f"{MODELS_DIR}/{coin}.keras"
    bak = f"{MODELS_DIR}/{coin}.keras.bak"

    if not os.path.exists(src):
        print(f"  ❌ {coin}: {src} not found")
        return False

    print(f"\n{'─'*55}")
    print(f"  Fixing {coin}...")

    # Backup original
    if not os.path.exists(bak):
        import shutil
        shutil.copy2(src, bak)
        print(f"    Backed up → {bak}")

    # Extract weights from old file
    try:
        raw_weights = extract_weights_from_keras(bak)
    except Exception as e:
        print(f"  ❌ {coin}: failed to extract weights: {e}")
        return False

    if not raw_weights:
        print(f"  ❌ {coin}: no weights found in archive")
        return False

    # Build fresh model with current Keras
    model = build_model(lookback=60)

    # Try smart shape-matching first, then ordered fallback
    ok = assign_weights(model, raw_weights)
    if not ok:
        print(f"    Shape-match failed, trying ordered assignment...")
        ok = assign_weights_ordered(model, raw_weights)

    if not ok:
        print(f"  ❌ {coin}: could not match weights — model needs retraining")
        return False

    # Save with current Keras version
    model.save(dst)
    print(f"  ✅ {coin}: saved fixed model → {dst}")

    # Quick verify — try loading it
    try:
        from tensorflow.keras.models import load_model
        m2 = load_model(dst, compile=False)
        dummy = np.zeros((1, 60, 1))
        out = m2.predict(dummy, verbose=0)
        print(f"    Verified: predict() output shape {out.shape}, value {out[0][0]:.6f}")
        return True
    except Exception as e:
        print(f"  ⚠️  {coin}: saved but verify failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=ALL_COINS)
    args = parser.parse_args()
    coins = [c.upper() for c in args.coins]

    print(f"\nFixing {len(coins)} model(s): {coins}\n")
    ok = fail = 0
    for coin in coins:
        if fix_coin(coin):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*55}")
    print(f"  Done: {ok} fixed, {fail} failed")
    if fail == 0:
        print("\n  Now test with:")
        print('  python -c "from src.lstm_engine import predict_coin; '
              'import json; print(json.dumps(predict_coin(\'BTC\'), indent=2))"')
    else:
        print("\n  Failed coins need retraining (retrain_local.py).")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()