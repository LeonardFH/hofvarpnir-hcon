"""
HÓFVARPNIRHCON — INFERENCE THROUGHPUT BENCHMARK
------------------------------------------------
Reads the combined dataset (229,409 molecules, deduplicated union of all six
datasets), splits molecules into 5 molecular weight bins, and runs the
inference pipeline separately on each bin to measure throughput as a
function of molecular weight.

Outputs:
    - Speed (molecules/second) per molecular weight bin
    - Total molecules processed
    - Wall time per bin

This script generates the data for Table 22 in the paper.
"""

import os
import sys
import time
import csv
import pickle
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import polars as pl
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

from hofvarpnirhcon_lib import calculate_molecule_properties_direct

RDLogger.DisableLog('rdApp.*')

# ============================================================
# CONFIG
# ============================================================
ROOT_FOLDER = "."                     # directory with your CSV files
WEIGHTS_FILE = "my_weights_235k.pkl"
MASTER_DIR = "MASTERHX"
N_CORES = 9
N_BINS = 5                            # number of MW groups

# ============================================================
# HELPER: molecular weight from SMILES
# ============================================================
def compute_mw(smiles):
    """Return molecular weight for a SMILES string, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)

# ============================================================
# CORE WORKER (adapted to process a single CSV chunk)
# ============================================================
def worker_for_chunk(core_id, chunk_csv, weights, output_dir, result_queue, bin_label):
    """
    Processes one chunk CSV (contains SMILES column).
    Writes a parquet file with the results.
    """
    from hofvarpnirhcon_lib import calculate_molecule_properties_direct

    try:
        Chem.SetNumThreads(1)
    except:
        pass

    # Use pre‑loaded weights
    def_overlap = weights.get('default_overlap', 5.9)
    dict_small = weights.get('small', {})
    dict_medium = weights.get('medium', {})
    dict_large = weights.get('large', {})

    t0 = time.perf_counter()
    processed = 0

    # Read the chunk
    try:
        df = pl.read_csv(chunk_csv)
        smiles_list = df['SMILES'].to_list()
    except Exception as e:
        print(f"[Core {core_id:02d}] ❌ Error reading {chunk_csv}: {e}")
        result_queue.put((core_id, 0, 0.0))
        return

    # Process each SMILES
    out_data = []
    for smi in smiles_list:
        if not smi:
            continue
        # Generate a dummy zinc_id (we only care about speed)
        zid = f"BIN_{bin_label}_C{core_id}_{processed:08d}"
        elem_str, pred_density = calculate_molecule_properties_direct(
            smi, dict_small, dict_medium, dict_large, def_overlap
        )
        if pred_density is None:
            continue
        out_data.append((zid, smi, elem_str, pred_density))
        processed += 1

    # Write a single parquet file for this chunk
    if out_data:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"bin_{bin_label}_chunk_{core_id:02d}.parquet")
        df_out = pl.DataFrame({
            "zinc_id": [d[0] for d in out_data],
            "SMILES": [d[1] for d in out_data],
            "Elements": [d[2] for d in out_data],
            "Predicted_Density": [d[3] for d in out_data],
        })
        df_out.write_parquet(out_file, compression="zstd")
        print(f"[Core {core_id:02d}] ✅ Wrote {out_file} ({processed} molecules)")

    elapsed = time.perf_counter() - t0
    speed = processed / elapsed if elapsed > 0 else 0
    print(f"[Core {core_id:02d}] 🏁 DONE. {processed} molecules in {elapsed:.2f}s → {speed:.0f} mol/s")
    result_queue.put((core_id, processed, elapsed))


# ============================================================
# PROCESS ONE BIN
# ============================================================
def process_bin(bin_label, smiles_list, weights, master_dir, n_cores):
    """
    Split the given SMILES list into chunks, run workers in parallel,
    return total molecules and wall time for this bin.
    """
    if not smiles_list:
        return 0, 0.0

    # Determine number of chunks (capped by number of SMILES)
    n_chunks = min(n_cores, len(smiles_list))
    if n_chunks == 0:
        return 0, 0.0

    # Create temporary CSV files for each chunk
    chunk_paths = []
    chunk_size = len(smiles_list) // n_chunks
    remainder = len(smiles_list) % n_chunks
    start = 0
    for i in range(n_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk_smiles = smiles_list[start:end]
        start = end
        # Write chunk to a temporary CSV
        chunk_file = os.path.join(master_dir, f"temp_bin{bin_label}_chunk{i}.csv")
        # Use polars to write quickly
        pl.DataFrame({"SMILES": chunk_smiles}).write_csv(chunk_file)
        chunk_paths.append(chunk_file)

    # Output directory for this bin
    bin_output_dir = os.path.join(master_dir, f"bin_{bin_label}")
    os.makedirs(bin_output_dir, exist_ok=True)

    # Launch workers
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    processes = []
    for core_id in range(n_chunks):
        p = ctx.Process(
            target=worker_for_chunk,
            args=(core_id, chunk_paths[core_id], weights, bin_output_dir, result_queue, bin_label)
        )
        processes.append(p)
        p.start()
        print(f"🔄 Bin {bin_label} core {core_id:02d} started (PID: {p.pid})")

    # Wait for all to finish
    for p in processes:
        p.join()

    # Collect results
    total_mol = 0
    total_time = 0.0
    while not result_queue.empty():
        core_id, mol, elapsed = result_queue.get_nowait()
        total_mol += mol
        if elapsed > total_time:
            total_time = elapsed   # wall time = max elapsed among cores

    # Clean up temporary chunk files
    for f in chunk_paths:
        try:
            os.remove(f)
        except:
            pass

    return total_mol, total_time


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    mp.freeze_support()
    script_start = time.perf_counter()

    print("=" * 70)
    print("🚀 ZINC22 SPEED TEST BY MOLECULAR WEIGHT BINS")
    print("=" * 70)
    print(f"📁 Root folder: {ROOT_FOLDER}")
    print(f"🧠 Weights: {WEIGHTS_FILE}")
    print(f"📁 Output: {MASTER_DIR}")
    print(f"🖥️  Cores: {N_CORES}")
    print(f"📊 Bins: {N_BINS}")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Load weights (once, shared)
    # ------------------------------------------------------------
    print("\n🧠 Loading weights...")
    with open(WEIGHTS_FILE, 'rb') as f:
        weights = pickle.load(f)
    print(f"✅ Weights loaded.")

    # ------------------------------------------------------------
    # 2. Read all CSV files and collect SMILES
    # ------------------------------------------------------------
    print("\n📂 Reading CSV files...")
    all_smiles = []
    for filename in os.listdir(ROOT_FOLDER):
        if filename.startswith("combined_all_datasets_dedup - Copy") and filename.endswith(".csv"):
            full_path = os.path.join(ROOT_FOLDER, filename)
            df = pl.read_csv(full_path)
            smiles_list = df['SMILES'].to_list()
            all_smiles.extend(smiles_list)
            print(f"  {filename}: {len(smiles_list)} molecules")

    total_input = len(all_smiles)
    print(f"✅ Total molecules read: {total_input:,}")

    # ------------------------------------------------------------
    # 3. Compute molecular weight for each SMILES
    # ------------------------------------------------------------
    print("\n⚖️ Computing molecular weights (this may take a moment)...")
    mw_list = []
    valid_smiles = []
    for smi in all_smiles:
        mw = compute_mw(smi)
        if mw is not None:
            mw_list.append(mw)
            valid_smiles.append(smi)
        # else discard (invalid SMILES)

    print(f"✅ Valid molecules: {len(valid_smiles):,} (discarded {total_input - len(valid_smiles):,})")

    # ------------------------------------------------------------
    # 4. Assign bins using quantiles (equal count per bin)
    # ------------------------------------------------------------
    if len(valid_smiles) == 0:
        print("❌ No valid molecules found. Exiting.")
        sys.exit(1)

    # Use pandas qcut for equal‑frequency bins
    mw_series = pd.Series(mw_list)
    try:
        # pd.qcut may fail if many duplicate values; we'll use np.percentile manually
        percentiles = np.linspace(0, 100, N_BINS + 1)[1:-1]
        bin_edges = [np.percentile(mw_list, p) for p in percentiles]
        # add -inf and +inf for boundaries
        bin_edges = [-np.inf] + bin_edges + [np.inf]
        labels = range(N_BINS)
        bin_indices = pd.cut(mw_series, bins=bin_edges, labels=labels, include_lowest=True)
    except Exception as e:
        print(f"⚠️ Quantile binning failed, falling back to equal‑width bins.")
        min_mw = min(mw_list)
        max_mw = max(mw_list)
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

    # ------------------------------------------------------------
    # 5. Run the pipeline for each bin sequentially
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🚀 Starting per‑bin speed tests...")
    print("=" * 70)

    bin_results = []
    for b in range(N_BINS):
        smiles_bin = bin_smiles[b]
        if not smiles_bin:
            print(f"\n⚠️ Bin {b} is empty, skipping.")
            continue

        print(f"\n🔥 Processing Bin {b} ({len(smiles_bin):,} molecules) ...")
        bin_start = time.perf_counter()
        mol_count, elapsed = process_bin(b, smiles_bin, weights, MASTER_DIR, N_CORES)
        speed = mol_count / elapsed if elapsed > 0 else 0
        bin_results.append((b, mol_count, elapsed, speed))
        print(f"✅ Bin {b} completed: {mol_count:,} molecules in {elapsed:.2f}s → {speed:,.0f} mol/s")

    # ------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 SPEED SUMMARY BY MOLECULAR WEIGHT BIN")
    print("=" * 70)
    print(f"{'Bin':<6} {'#Molecules':>12} {'Time (s)':>10} {'Speed (mol/s)':>15}")
    print("-" * 50)
    for b, cnt, t, spd in sorted(bin_results):
        print(f"{b:<6} {cnt:>12,} {t:>10.2f} {spd:>15,.0f}")
    print("=" * 70)
    print(f"📁 Results written to subdirectories under {MASTER_DIR}/")
    print(f"⏱️  Total script time: {time.perf_counter() - script_start:.2f}s")
    print("=" * 70)