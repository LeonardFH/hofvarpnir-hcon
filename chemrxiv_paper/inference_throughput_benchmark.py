"""
HÓFVARPNIRHCON — INFERENCE THROUGHPUT BENCHMARK (CONFIGURATION‑DRIVEN)
----------------------------------------------------------------------
Set the configuration variables below, then run the script.
No command‑line arguments, no manual file splitting.
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

# Import the prediction function (adjust to your package as needed)
# If you are using the new package, replace with:
#   from hofvarpnirhcon import predict_density_batch
# But this script uses the fast per‑molecule function for maximum throughput.
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
# MAIN
# ============================================================
if __name__ == "__main__":
    script_start = time.perf_counter()

    print("=" * 70)
    print("🚀 INFERENCE THROUGHPUT BENCHMARK (CONFIG‑DRIVEN)")
    print("=" * 70)
    print(f"📁 Input CSV:   {INPUT_CSV}")
    print(f"🧠 Weights:     {WEIGHTS_FILE}")
    print(f"🖥️  Cores:       {N_CORES}")
    print(f"📦 Chunk size:  {'auto (per core)' if CHUNK_SIZE == 0 else CHUNK_SIZE}")
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
    # 3. Compute molecular weights (optional, but we keep for binning later)
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
    # 4. Split into chunks
    # ------------------------------------------------------------
    if CHUNK_SIZE == 0:
        # Auto‑split by number of cores
        n_chunks = min(N_CORES, total_valid)
        chunk_size = total_valid // n_chunks
        remainder = total_valid % n_chunks
        chunk_sizes = [chunk_size + (1 if i < remainder else 0) for i in range(n_chunks)]
    else:
        # Fixed chunk size
        n_chunks = (total_valid + CHUNK_SIZE - 1) // CHUNK_SIZE
        chunk_sizes = [min(CHUNK_SIZE, total_valid - i * CHUNK_SIZE) for i in range(n_chunks)]

    print(f"\n📦 Splitting into {n_chunks} chunks...")
    temp_dir = tempfile.mkdtemp(prefix="hcon_benchmark_")
    chunk_paths = []

    start_idx = 0
    for i, size in enumerate(chunk_sizes):
        end_idx = start_idx + size
        chunk_smiles = valid_smiles[start_idx:end_idx]
        csv_path = os.path.join(temp_dir, f"chunk_{i:03d}.csv")
        pl.DataFrame({"SMILES": chunk_smiles}).write_csv(csv_path)
        chunk_paths.append(csv_path)
        if VERBOSE:
            print(f"  Chunk {i:03d}: {len(chunk_smiles):,} molecules -> {os.path.basename(csv_path)}")
        start_idx = end_idx

    # ------------------------------------------------------------
    # 5. Launch workers
    # ------------------------------------------------------------
    print(f"\n🚀 Launching {n_chunks} workers...\n")
    result_queue = Queue()
    processes = []

    for core_id, csv_path in enumerate(chunk_paths):
        p = Process(
            target=worker_for_chunk,
            args=(core_id, csv_path, weights, result_queue)
        )
        processes.append(p)
        p.start()
        if VERBOSE:
            print(f"  Core {core_id:02d} started (PID: {p.pid})")

    # ------------------------------------------------------------
    # 6. Collect results
    # ------------------------------------------------------------
    all_results = []
    total_mol = 0
    total_time = 0.0

    for _ in processes:
        core_id, mol, elapsed, res = result_queue.get()
        all_results.extend(res)
        total_mol += mol
        if elapsed > total_time:
            total_time = elapsed

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # ------------------------------------------------------------
    # 7. Clean up temporary files
    # ------------------------------------------------------------
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n🧹 Cleaned up temporary files.")

    # ------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------
    speed = total_mol / total_time if total_time > 0 else 0
    print("\n" + "=" * 70)
    print("📊 SPEED SUMMARY")
    print("=" * 70)
    print(f"  Total molecules:     {total_mol:,}")
    print(f"  Wall time:           {total_time:.2f}s")
    print(f"  Throughput:          {speed:,.0f} mol/s")
    print(f"  Per‑core throughput: {speed / n_chunks:,.0f} mol/s/core")
    print("=" * 70)

    # (Optional) You can add MW binning here if you want the per‑bin breakdown.
    # The earlier version had that; I can add it back if desired.

    print(f"\n⏱️  Total script time: {time.perf_counter() - script_start:.2f}s")
    print("=" * 70)