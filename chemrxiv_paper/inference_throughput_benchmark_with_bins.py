"""
HÓFVARPNIRHCON — INFERENCE THROUGHPUT BENCHMARK WITH MW BINNING
----------------------------------------------------------------
Set the configuration variables below, then run the script.
Optionally splits molecules into MW bins and reports speed per bin.
"""

import os
import sys
import time
import pickle
import tempfile
import shutil
from collections import defaultdict

import polars as pl
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from multiprocessing import Process, Queue

# ------------------------------------------------------------------
# 🔧 CONFIGURATION — SET THESE VARIABLES BEFORE RUNNING
# ------------------------------------------------------------------

# Path to your input CSV file (must contain a 'SMILES' column)
INPUT_CSV = "combined_all_datasets_dedup.csv"

# Path to your trained weights .pkl file
WEIGHTS_FILE = "my_weights_235k.pkl"

# Number of CPU cores to use
N_CORES = 9

# Molecules per chunk (0 = auto‑split by number of cores)
CHUNK_SIZE = 0

# Show per‑core progress (True/False)
VERBOSE = True

# ------------------------------------------------------------------
# MW BINNING CONFIGURATION
# ------------------------------------------------------------------

# Number of MW bins (0 = disable binning, run aggregate only)
N_BINS = 5

# ------------------------------------------------------------------

# Import the prediction function (adjust to your package as needed)
from hofvarpnirhcon_lib import calculate_molecule_properties_direct

RDLogger.DisableLog('rdApp.*')


# ============================================================
# HELPER: molecular weight from SMILES
# ============================================================
def compute_mw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)


# ============================================================
# WORKER: processes one CSV chunk
# ============================================================
def worker_for_chunk(core_id, chunk_csv, weights, result_queue):
    """Process one chunk, return (core_id, processed_count, elapsed, list_of_densities)."""
    from hofvarpnirhcon_lib import calculate_molecule_properties_direct
    try:
        Chem.SetNumThreads(1)
    except:
        pass

    def_overlap = weights.get('default_overlap', 5.9)
    dict_small = weights.get('small', {})
    dict_medium = weights.get('medium', {})
    dict_large = weights.get('large', {})

    t0 = time.perf_counter()
    processed = 0
    results = []

    try:
        df = pl.read_csv(chunk_csv)
        smiles_list = df['SMILES'].to_list()
    except Exception as e:
        print(f"[Core {core_id:02d}] ❌ Error reading {chunk_csv}: {e}")
        result_queue.put((core_id, 0, 0.0, []))
        return

    for smi in smiles_list:
        if not smi:
            continue
        _, pred_density = calculate_molecule_properties_direct(
            smi, dict_small, dict_medium, dict_large, def_overlap
        )
        results.append(pred_density)
        if pred_density is not None:
            processed += 1

    elapsed = time.perf_counter() - t0
    if VERBOSE:
        speed = processed / elapsed if elapsed > 0 else 0
        print(f"[Core {core_id:02d}] 🏁 {processed} molecules in {elapsed:.2f}s → {speed:.0f} mol/s")
    result_queue.put((core_id, processed, elapsed, results))


# ============================================================
# RUN PARALLEL ON A LIST OF SMILES
# ============================================================
def run_parallel(smiles_list, label, weights, n_cores, chunk_size):
    """
    Run the inference pipeline on a list of SMILES in parallel.
    Returns (processed_count, wall_time, list_of_densities).
    """
    if not smiles_list:
        return 0, 0.0, []

    total = len(smiles_list)

    # Determine chunking
    if chunk_size == 0:
        n_chunks = min(n_cores, total)
        chunk_sizes = [total // n_chunks + (1 if i < total % n_chunks else 0) for i in range(n_chunks)]
    else:
        n_chunks = (total + chunk_size - 1) // chunk_size
        chunk_sizes = [min(chunk_size, total - i * chunk_size) for i in range(n_chunks)]

    # Create temporary CSV chunks
    temp_dir = tempfile.mkdtemp(prefix="hcon_benchmark_")
    chunk_paths = []
    start_idx = 0
    for i, size in enumerate(chunk_sizes):
        end_idx = start_idx + size
        chunk_smiles = smiles_list[start_idx:end_idx]
        csv_path = os.path.join(temp_dir, f"chunk_{i:03d}.csv")
        pl.DataFrame({"SMILES": chunk_smiles}).write_csv(csv_path)
        chunk_paths.append(csv_path)
        if VERBOSE:
            print(f"  [{label}] Chunk {i:03d}: {len(chunk_smiles):,} molecules")
        start_idx = end_idx

    # Launch workers
    result_queue = Queue()
    processes = []
    for core_id, csv_path in enumerate(chunk_paths):
        p = Process(
            target=worker_for_chunk,
            args=(core_id, csv_path, weights, result_queue)
        )
        processes.append(p)
        p.start()

    # Collect results
    all_results = []
    total_mol = 0
    total_time = 0.0

    for _ in processes:
        core_id, mol, elapsed, res = result_queue.get()
        all_results.extend(res)
        total_mol += mol
        if elapsed > total_time:
            total_time = elapsed

    for p in processes:
        p.join()

    # Clean up
    shutil.rmtree(temp_dir, ignore_errors=True)

    return total_mol, total_time, all_results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    script_start = time.perf_counter()

    print("=" * 70)
    print("🚀 INFERENCE THROUGHPUT BENCHMARK WITH MW BINNING")
    print("=" * 70)
    print(f"📁 Input CSV:   {INPUT_CSV}")
    print(f"🧠 Weights:     {WEIGHTS_FILE}")
    print(f"🖥️  Cores:       {N_CORES}")
    print(f"📦 Chunk size:  {'auto (per core)' if CHUNK_SIZE == 0 else CHUNK_SIZE}")
    print(f"📊 MW bins:     {N_BINS if N_BINS > 0 else 'disabled'}")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Load weights
    # ------------------------------------------------------------
    print("\n🧠 Loading weights...")
    with open(WEIGHTS_FILE, 'rb') as f:
        weights = pickle.load(f)
    print("✅ Weights loaded.")

    # ------------------------------------------------------------
    # 2. Read input CSV
    # ------------------------------------------------------------
    print(f"\n📂 Reading input CSV: {INPUT_CSV}")
    try:
        df = pl.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        sys.exit(1)

    if 'SMILES' not in df.columns:
        print("❌ Input CSV must contain a 'SMILES' column.")
        sys.exit(1)

    all_smiles = df['SMILES'].to_list()
    total_input = len(all_smiles)
    print(f"✅ Total molecules read: {total_input:,}")

    # ------------------------------------------------------------
    # 3. Compute molecular weights
    # ------------------------------------------------------------
    print("\n⚖️ Computing molecular weights...")
    mw_list = []
    valid_smiles = []
    for smi in all_smiles:
        mw = compute_mw(smi)
        if mw is not None:
            mw_list.append(mw)
            valid_smiles.append(smi)

    total_valid = len(valid_smiles)
    print(f"✅ Valid molecules: {total_valid:,} (discarded {total_input - total_valid:,})")

    if total_valid == 0:
        print("❌ No valid molecules found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------
    # 4. Assign MW bins (if enabled)
    # ------------------------------------------------------------
    if N_BINS > 0:
        mw_series = pd.Series(mw_list)
        try:
            percentiles = np.linspace(0, 100, N_BINS + 1)[1:-1]
            bin_edges = [np.percentile(mw_list, p) for p in percentiles]
            bin_edges = [-np.inf] + bin_edges + [np.inf]
            bin_indices = pd.cut(mw_series, bins=bin_edges, labels=range(N_BINS), include_lowest=True)
        except Exception as e:
            print(f"⚠️ Quantile binning failed, falling back to equal‑width bins.")
            min_mw, max_mw = min(mw_list), max(mw_list)
            width = (max_mw - min_mw) / N_BINS
            bin_edges = [min_mw + i * width for i in range(N_BINS + 1)]
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            bin_indices = pd.cut(mw_series, bins=bin_edges, labels=range(N_BINS), include_lowest=True)

        # Group SMILES by bin
        bin_smiles = defaultdict(list)
        for idx, smi in enumerate(valid_smiles):
            bin_id = bin_indices.iloc[idx]
            bin_smiles[bin_id].append(smi)

        # Print bin statistics
        print("\n📊 Molecular weight bins:")
        for b in range(N_BINS):
            cnt = len(bin_smiles[b])
            if cnt > 0:
                mw_vals = [mw_list[i] for i in range(len(mw_list)) if bin_indices.iloc[i] == b]
                print(f"  Bin {b}: {cnt:,} molecules, MW range {min(mw_vals):.1f} – {max(mw_vals):.1f} Da")
    else:
        # No binning: treat all as one bin
        bin_smiles = {0: valid_smiles}

    # ------------------------------------------------------------
    # 5. Run the pipeline for each bin (sequentially)
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🚀 Running inference per bin...")
    print("=" * 70)

    bin_results = []
    total_mol_all = 0
    total_time_all = 0.0

    for b in sorted(bin_smiles.keys()):
        smiles_bin = bin_smiles[b]
        if not smiles_bin:
            continue

        label = f"Bin {b}" if N_BINS > 0 else "Aggregate"
        print(f"\n🔥 Processing {label} ({len(smiles_bin):,} molecules) ...")

        bin_start = time.perf_counter()
        mol_count, elapsed, _ = run_parallel(
            smiles_bin,
            label,
            weights,
            n_cores=N_CORES,
            chunk_size=CHUNK_SIZE
        )
        speed = mol_count / elapsed if elapsed > 0 else 0
        bin_results.append((b, mol_count, elapsed, speed, label))
        total_mol_all += mol_count
        total_time_all += elapsed  # sequential sum

        print(f"✅ {label} completed: {mol_count:,} molecules in {elapsed:.2f}s → {speed:,.0f} mol/s")

    # ------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 SPEED SUMMARY BY MW BIN")
    print("=" * 70)
    print(f"{'Bin':<8} {'#Molecules':>12} {'Time (s)':>10} {'Speed (mol/s)':>15}")
    print("-" * 50)

    for b, cnt, t, spd, label in bin_results:
        if N_BINS > 0:
            bin_label = f"Bin {b}"
        else:
            bin_label = "All"
        print(f"{bin_label:<8} {cnt:>12,} {t:>10.2f} {spd:>15,.0f}")

    print("=" * 70)

    # Aggregate throughput (total molecules / total sequential wall time)
    agg_speed = total_mol_all / total_time_all if total_time_all > 0 else 0
    print(f"  TOTAL (sequential sum): {total_mol_all:,} molecules in {total_time_all:.2f}s → {agg_speed:,.0f} mol/s")
    print("=" * 70)

    print(f"\n⏱️  Total script time: {time.perf_counter() - script_start:.2f}s")
    print("=" * 70)