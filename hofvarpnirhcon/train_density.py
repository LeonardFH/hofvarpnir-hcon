"""
Train density dictionaries from CSV data
With auto-detect MW strategy + user override
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import nnls
from tqdm import tqdm
import pickle
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ============================================================================
# CONSTANTS
# ============================================================================
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
MAGIC_SCALE = PI * PHI
DEFAULT_OVERLAP = 0.3
MW_SMALL_MAX = 180
MW_MEDIUM_MAX = 400
MIN_PER_CLASS = 150

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return sum(atom.GetMass() for atom in mol.GetAtoms())

def magic_volume(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    total_cuberoot = sum(atom.GetMass() ** (1/3) for atom in mol.GetAtoms())
    return MAGIC_SCALE * total_cuberoot

def get_bond_type_only(smiles):
    """Get bond types without lengths."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        
        bonds = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetSymbol()
            a2 = bond.GetEndAtom().GetSymbol()
            bt = bond.GetBondType()
            order = 1 if bt == Chem.rdchem.BondType.SINGLE else 2 if bt == Chem.rdchem.BondType.DOUBLE else 3
            
            if a1 > a2:
                a1, a2 = a2, a1
            bond_key = tuple(sorted([a1, a2])) + (order,)
            bonds.append(bond_key)
        return bonds
    except:
        return None

def train_type_dictionary(data, min_bond_count=2):
    """Train dictionary using bond types only (NNLS)."""
    if len(data) < 50:
        return {}
    
    bond_counts = {}
    for row in data:
        for b in row['bonds_type_only']:
            bond_counts[b] = bond_counts.get(b, 0) + 1
    
    bond_types = [b for b, c in bond_counts.items() if c >= min_bond_count]
    if len(bond_types) == 0:
        return {}
    
    X = []
    y = []
    for row in data:
        target = row['V_magic'] - row['V_exp']
        counts = {bt: 0 for bt in bond_types}
        for b in row['bonds_type_only']:
            if b in counts:
                counts[b] += 1
        X.append([counts[bt] for bt in bond_types])
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    overlaps, _ = nnls(X, y)
    
    return {bt: ov for bt, ov in zip(bond_types, overlaps) if ov > 0}

def determine_mw_strategy(data, min_per_class=150):
    total = len(data)
    
    small = sum(1 for d in data if d['mw'] < MW_SMALL_MAX)
    medium = sum(1 for d in data if MW_SMALL_MAX <= d['mw'] < MW_MEDIUM_MAX)
    large = sum(1 for d in data if d['mw'] >= MW_MEDIUM_MAX)
    below_400 = small + medium
    
    details = {
        'total': total,
        'small': small,
        'medium': medium,
        'large': large,
        'below_400': below_400,
        'above_400': large,
    }
    
    if total < 50:
        raise ValueError(f"Too few molecules ({total}). Need at least 50 for training.")
    
    if total < 150:
        return 'single', None, details
    
    if small >= min_per_class and medium >= min_per_class and large >= min_per_class:
        return 'three_class', {'small_max': MW_SMALL_MAX, 'medium_max': MW_MEDIUM_MAX}, details
    
    if below_400 >= min_per_class and large >= min_per_class:
        return 'two_class', {'split': MW_MEDIUM_MAX}, details
    
    if small >= min_per_class and (medium + large) >= min_per_class:
        return 'two_class', {'split': MW_SMALL_MAX}, details
    
    return 'single', None, details

def build_class_data(data, strategy, config):
    """Split data into classes based on strategy."""
    if strategy == 'single':
        return {'global': data}
    
    elif strategy == 'two_class':
        split = config['split']
        if split == MW_MEDIUM_MAX:  # <400, >=400
            # 🔥 FIX: Keep small and medium separate for training
            small = [d for d in data if d['mw'] < MW_SMALL_MAX]
            medium = [d for d in data if MW_SMALL_MAX <= d['mw'] < MW_MEDIUM_MAX]
            large = [d for d in data if d['mw'] >= MW_MEDIUM_MAX]
            return {'small': small, 'medium': medium, 'large': large}
        else:  # <180, >=180
            class1 = [d for d in data if d['mw'] < MW_SMALL_MAX]
            class2 = [d for d in data if d['mw'] >= MW_SMALL_MAX]
            return {'small': class1, 'large': class2}
    
    elif strategy == 'three_class':
        small = [d for d in data if d['mw'] < MW_SMALL_MAX]
        medium = [d for d in data if MW_SMALL_MAX <= d['mw'] < MW_MEDIUM_MAX]
        large = [d for d in data if d['mw'] >= MW_MEDIUM_MAX]
        return {'small': small, 'medium': medium, 'large': large}
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def generate_output_filename(data_path, strategy, config):
    base = os.path.splitext(os.path.basename(data_path))[0]
    
    if strategy == 'single':
        suffix = 'global'
    elif strategy == 'two_class':
        split = config.get('split', 400)
        suffix = f'twoclass_{split}'
    else:
        suffix = 'threeclass'
    
    return f"{base}_{suffix}.pkl"

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_density(
    data_path: str,
    output_path: str = None,
    strategy: str = 'auto',
    min_per_class: int = 150,
    filter_hcon: bool = True,
    filter_cocrystals: bool = True,  # 👈 NEW PARAMETER
    verbose: bool = True
):
    """
    Train density dictionaries from CSV data.
    
    Args:
        data_path: Path to CSV file with 'SMILES' and 'Density' columns
        output_path: Output .pkl file path (auto-generated if None)
        strategy: 'auto', 'single', 'two', or 'three'
        min_per_class: Minimum molecules per class for splitting (default: 150)
        filter_hcon: If True, only keep H/C/N/O molecules
        filter_cocrystals: If True, exclude cocrystals (SMILES with '.') from training
        verbose: Print progress if True
    
    Returns:
        dict: Trained weights with metadata
    """
    if verbose:
        print("=" * 80)
        print("TRAIN DENSITY DICTIONARIES")
        print("=" * 80)
        print(f"\n[1/5] Loading data from: {data_path}")
    
    df = pd.read_csv(data_path).dropna(subset=["SMILES", "Density"])
        
    # Filter for HCON only (optional)
    if filter_hcon:
        ALLOWED = {"H", "C", "N", "O"}
        def is_hcon(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            return all(atom.GetSymbol() in ALLOWED for atom in mol.GetAtoms())
        
        df = df[df["SMILES"].apply(is_hcon)].reset_index(drop=True)
        if verbose:
            print(f"  Filtered H/C/N/O molecules: {len(df):,}")

    # 🔥 NEW: Filter out cocrystals (SMILES with '.')
    n_coco = 0  # 👈 ADD THIS LINE — ensures n_coco is always defined
    if filter_cocrystals:
        cocrystal_mask = df["SMILES"].str.contains(r'\.', na=False, regex=True)
        n_coco = cocrystal_mask.sum()
        if n_coco > 0:
            df = df[~cocrystal_mask].copy()
            if verbose:
                print(f"  Filtered out cocrystals: {n_coco:,} molecules removed")
                print(f"  Pure molecules remaining: {len(df):,}")

    if verbose:
        print("\n[2/5] Building dataset...")
    
    data = []
    failed = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing", disable=not verbose):
        smiles = row["SMILES"]
        exp = row["Density"]
        
        mw = get_molecular_weight(smiles)
        if mw is None:
            failed += 1
            continue
        
        V_magic = magic_volume(smiles)
        if V_magic is None:
            failed += 1
            continue
        
        bonds_type_only = get_bond_type_only(smiles)
        if bonds_type_only is None:
            failed += 1
            continue
        
        V_exp = mw / exp
        
        data.append({
            'smiles': smiles,
            'exp': exp,
            'mw': mw,
            'V_magic': V_magic,
            'V_exp': V_exp,
            'bonds_type_only': bonds_type_only,
        })
    
    if verbose:
        print(f"  Valid molecules: {len(data):,}")
        print(f"  Failed: {failed}")
    
    if len(data) < 50:
        raise ValueError(f"Too few molecules ({len(data)}). Need at least 50 for training.")
    
    if verbose:
        print("\n[3/5] Determining MW strategy...")
    
    auto_strategy, auto_config, details = determine_mw_strategy(data, min_per_class)
    
    if verbose:
        print(f"\n  Dataset size: {details['total']} molecules")
        print(f"  MW distribution:")
        print(f"    Small (<180):  {details['small']}")
        print(f"    Medium (180-400): {details['medium']}")
        print(f"    Large (>400):   {details['large']}")
    
    if strategy == 'auto':
        final_strategy = auto_strategy
        config = auto_config
        if verbose:
            print(f"\n  Auto-detected strategy: {final_strategy.upper()}")
    else:
        strategy_map = {
            'single': 'single',
            'two': 'two_class',
            'three': 'three_class',
        }
        final_strategy = strategy_map[strategy]
        config = None
        
        if final_strategy == 'three_class':
            config = {'small_max': MW_SMALL_MAX, 'medium_max': MW_MEDIUM_MAX}
        elif final_strategy == 'two_class':
            config = {'split': MW_MEDIUM_MAX}
        
        if verbose:
            print(f"\n  User override: {final_strategy.upper()}")
        
        if final_strategy == 'three_class' and details['total'] < 300:
            if verbose:
                print(f"  ⚠️ WARNING: Three classes with only {details['total']} molecules may be unstable.")
                print(f"     Consider using strategy='single' or 'two'.")
        elif final_strategy == 'two_class' and details['total'] < 150:
            if verbose:
                print(f"  ⚠️ WARNING: Two classes with only {details['total']} molecules may be unstable.")
                print(f"     Consider using strategy='single'.")
    
    if verbose:
        print("\n[4/5] Training dictionaries...")
    
    class_data = build_class_data(data, final_strategy, config)
    weights = {}
    
    for class_name, class_items in class_data.items():
        if len(class_items) == 0:
            if verbose:
                print(f"  {class_name}: 0 molecules — skipping")
            weights[class_name] = {}
            continue
        
        if verbose:
            print(f"  {class_name}: {len(class_items)} molecules")
        dict_result = train_type_dictionary(class_items)
        weights[class_name] = dict_result
        if verbose:
            print(f"    → {len(dict_result)} bond types")
    
    # ========================================================================
    # 🔥 FIX: Merge small + medium for two_class (matching original script)
    # ========================================================================
    if final_strategy == 'two_class' and 'small' in weights and 'medium' in weights:
        if verbose:
            print(f"\n  Merging small + medium for two-class prediction...")
        weights['medium'] = {**weights['small'], **weights['medium']}
        if verbose:
            print(f"    → Merged dictionary: {len(weights['medium'])} bond types")
    
    if verbose:
        print("\n[5/5] Saving weights...")
    
    if output_path is None:
        output_path = generate_output_filename(data_path, final_strategy, config)
    elif not output_path.endswith('.pkl'):
        output_path += '.pkl'
    
    weights['metadata'] = {
        'strategy': final_strategy,
        'config': config,
        'mw_cutoffs': {'small_max': MW_SMALL_MAX, 'medium_max': MW_MEDIUM_MAX},
        'n_molecules': details['total'],
        'n_small': details['small'],
        'n_medium': details['medium'],
        'n_large': details['large'],
        'n_cocrystals_filtered': n_coco if filter_cocrystals else 0,  # 👈 Track filtered cocrystals
        'date_trained': datetime.now().isoformat(),
        'source_data': os.path.basename(data_path),
        'min_per_class': min_per_class,
        'filter_cocrystals': filter_cocrystals,
    }
    weights['default_overlap'] = DEFAULT_OVERLAP
    
    with open(output_path, 'wb') as f:
        pickle.dump(weights, f)
    
    if verbose:
        print(f"  ✅ Weights saved to: {output_path}")
    
    if verbose:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\n  Strategy:      {final_strategy.upper()}")
        print(f"  Output file:   {output_path}")
        print(f"  Molecules:     {details['total']}")
        print(f"  Bond types:    {sum(len(w) for k, w in weights.items() if k not in ['metadata', 'default_overlap'])}")
        if filter_cocrystals and n_coco > 0:
            print(f"  Cocrystals filtered: {n_coco:,}")
        print("\n" + "=" * 80)
    
    return weights
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train density dictionaries from CSV data"
    )
    parser.add_argument('--data', required=True,
                       help='CSV file with SMILES and Density columns')
    parser.add_argument('--output', default=None,
                       help='Output .pkl file (default: auto-generated from input filename)')
    parser.add_argument('--strategy', choices=['auto', 'single', 'two', 'three'],
                       default='auto',
                       help='MW splitting strategy (default: auto-detect)')
    parser.add_argument('--min-per-class', type=int, default=150,
                       help=f'Minimum molecules per class for splitting (default: 150)')
    parser.add_argument('--no-filter', action='store_true',
                       help='Disable H/C/N/O filtering')
    parser.add_argument('--keep-cocrystals', action='store_true',
                       help='Keep cocrystals in training data (default: filter them out)')
    
    args = parser.parse_args()
    
    train_density(
        data_path=args.data,
        output_path=args.output,
        strategy=args.strategy,
        min_per_class=args.min_per_class,
        filter_hcon=not args.no_filter,
        filter_cocrystals=not args.keep_cocrystals,  # 👈 Default: filter cocrystals
        verbose=True
    )