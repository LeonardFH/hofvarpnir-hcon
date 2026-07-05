"""
HófvarpnirHCON : Fast dictionary-based crystal density prediction

Fast dictionary-based model (no conformer generation, no voxel grids).
Supports pure crystals and cocrystals with MW-split dictionaries.
"""

import numpy as np
import pickle
import os
from rdkit import Chem
from typing import Union, List, Dict, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# ============================================================================
# CONSTANTS
# ============================================================================
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
MAGIC_SCALE = PI * PHI
DEFAULT_OVERLAP = 0.3
MW_SMALL_MAX = 180
MW_MEDIUM_MAX = 400
G_FACTOR = 1.660539
DEFAULT_CORES = 4
DEFAULT_CHUNK_SIZE = 4096

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_key(atom1: str, atom2: str, bond_order: int) -> tuple:
    """Create alphabetically sorted bond key for dictionary lookup."""
    return tuple(sorted([atom1, atom2])) + (bond_order,)

def get_molecular_weight(smiles: str) -> Optional[float]:
    """Get molecular weight from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return sum(atom.GetMass() for atom in mol.GetAtoms())

def magic_volume(smiles: str) -> Optional[float]:
    """
    Calculate magic volume: π·φ·Σ(mass^(1/3))
    This is the ideal packing volume before bond overlap corrections.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    total_cuberoot = sum(atom.GetMass() ** (1/3) for atom in mol.GetAtoms())
    return MAGIC_SCALE * total_cuberoot
    
def get_atom_counts(smiles: str) -> Optional[Dict[str, int]]:
    """Get atom counts for C, H, O, N."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    counts = {'C': 0, 'H': 0, 'O': 0, 'N': 0}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
    return counts    

def get_bond_overlap(atom1: str, atom2: str, bond_order: int, overlaps_dict: dict, default_overlap: float) -> float:
    """Get bond overlap from dictionary, return default_overlap if missing."""
    key = make_key(atom1, atom2, bond_order)
    return overlaps_dict.get(key, default_overlap)

def get_overlaps_for_molecule(
    smiles: str, 
    mw: float, 
    weights: dict,
    strategy: str
) -> tuple:
    """
    Return the appropriate overlap dictionary based on strategy and MW.
    
    Returns:
        (overlaps_dict, class_name, fallback_used)
    """
    default_overlap = weights.get('default_overlap', DEFAULT_OVERLAP)
    
    # Cocrystal detection
    if '.' in smiles:
        # For cocrystals, use global dictionary if available, else fallback
        if 'global' in weights:
            return weights['global'], 'global', False
        elif 'medium' in weights:  # Use medium as fallback
            return weights['medium'], 'medium', True
        else:
            # Use first available dict
            for key in ['small', 'medium', 'large']:
                if key in weights and weights[key]:
                    return weights[key], key, True
            return {}, 'fallback', True
    
    # Strategy-based selection
    if strategy == 'single':
        # Use global dictionary
        if 'global' in weights and weights['global']:
            return weights['global'], 'global', False
        else:
            # Fallback to any available dict
            for key in ['small', 'medium', 'large']:
                if key in weights and weights[key]:
                    return weights[key], key, True
            return {}, 'fallback', True
    
    elif strategy == 'two_class':
        split = weights.get('metadata', {}).get('config', {}).get('split', MW_MEDIUM_MAX)
        
        if mw < split:
            # CRITICAL FIX: Merge small + medium for two_class (matching standalone script)
            merged_dict = {}
            if 'small' in weights:
                merged_dict.update(weights['small'])
            if 'medium' in weights:
                merged_dict.update(weights['medium'])
            
            if merged_dict:
                return merged_dict, 'medium', False
            elif 'small' in weights and weights['small']:
                return weights['small'], 'small', False
            else:
                return weights.get('global', {}), 'global', True
        else:
            # Use large class
            if 'large' in weights and weights['large']:
                return weights['large'], 'large', False
            else:
                return weights.get('global', {}), 'global', True
    
    elif strategy == 'three_class':
        if mw < MW_SMALL_MAX:
            if 'small' in weights and weights['small']:
                return weights['small'], 'small', False
            else:
                # Fallback to medium or global
                if 'medium' in weights and weights['medium']:
                    return weights['medium'], 'medium', True
                else:
                    return weights.get('global', {}), 'global', True
        elif mw < MW_MEDIUM_MAX:
            if 'medium' in weights and weights['medium']:
                return weights['medium'], 'medium', False
            else:
                # Fallback to small or large
                if 'small' in weights and weights['small']:
                    return weights['small'], 'small', True
                elif 'large' in weights and weights['large']:
                    return weights['large'], 'large', True
                else:
                    return weights.get('global', {}), 'global', True
        else:
            if 'large' in weights and weights['large']:
                return weights['large'], 'large', False
            else:
                # Fallback to medium or global
                if 'medium' in weights and weights['medium']:
                    return weights['medium'], 'medium', True
                else:
                    return weights.get('global', {}), 'global', True
    
    else:
        # Unknown strategy — try global, then any dict
        if 'global' in weights and weights['global']:
            return weights['global'], 'global', False
        for key in ['small', 'medium', 'large']:
            if key in weights and weights[key]:
                return weights[key], key, True
        return {}, 'fallback', True

def predict_single_molecule(
    smiles: str, 
    overlaps_dict: dict, 
    default_overlap: float = DEFAULT_OVERLAP
) -> Optional[float]:
    """
    Predict density for a single molecule using dictionary.
    Returns density only (float).
    """
    mw = get_molecular_weight(smiles)
    if mw is None:
        return None
    
    V_magic = magic_volume(smiles)
    if V_magic is None:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    
    total_overlap = 0.0
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetSymbol()
        a2 = bond.GetEndAtom().GetSymbol()
        bt = bond.GetBondType()
        
        if bt == Chem.rdchem.BondType.SINGLE:
            order = 1
        elif bt == Chem.rdchem.BondType.DOUBLE:
            order = 2
        elif bt == Chem.rdchem.BondType.TRIPLE:
            order = 3
        else:
            order = 3
        
        overlap = get_bond_overlap(a1, a2, order, overlaps_dict, default_overlap)
        total_overlap += overlap
    
    V_corrected = V_magic - total_overlap
    if V_corrected <= 0:
        V_corrected = V_magic * 0.5
    
    return mw / V_corrected

# ============================================================================
# FAST PREDICTION HELPERS (for multiprocessing)
# ============================================================================

def _predict_single_with_dicts_fast(
    smiles: str,
    dict_small: dict,
    dict_medium: dict,
    dict_large: dict,
    dict_coco: dict,
    default_overlap: float = DEFAULT_OVERLAP
) -> Optional[float]:
    """
    Fast single prediction — direct dictionary selection (no strategy logic).
    This is the speed-critical path for batch predictions.
    """
    mw = get_molecular_weight(smiles)
    if mw is None:
        return None
    
    V_magic = magic_volume(smiles)
    if V_magic is None:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    
    # Direct dictionary selection (no strategy logic)
    if '.' in smiles:
        overlaps_dict = dict_coco
    elif mw < MW_SMALL_MAX:
        overlaps_dict = dict_small
    elif mw < MW_MEDIUM_MAX:
        overlaps_dict = dict_medium
    else:
        overlaps_dict = dict_large
    
    if overlaps_dict is None or len(overlaps_dict) == 0:
        return None
    
    total_overlap = 0.0
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetSymbol()
        a2 = bond.GetEndAtom().GetSymbol()
        bt = bond.GetBondType()
        order = 1 if bt == Chem.rdchem.BondType.SINGLE else 2 if bt == Chem.rdchem.BondType.DOUBLE else 3
        key = make_key(a1, a2, order)
        total_overlap += overlaps_dict.get(key, default_overlap)
    
    V_corrected = V_magic - total_overlap
    if V_corrected <= 0:
        V_corrected = V_magic * 0.5
    
    return mw / V_corrected


def _predict_batch_chunk_fast(
    chunk: List[str],
    dict_small: dict,
    dict_medium: dict,
    dict_large: dict,
    dict_coco: dict,
    default_overlap: float = DEFAULT_OVERLAP
) -> List[Optional[float]]:
    """
    Predict a chunk of SMILES using fast path.
    """
    results = []
    for smiles in chunk:
        result = _predict_single_with_dicts_fast(
            smiles, dict_small, dict_medium, dict_large, dict_coco, default_overlap
        )
        results.append(result)
    return results

# ============================================================================
# MAIN PREDICTION FUNCTIONS
# ============================================================================

def predict_density(
    smiles: str,
    weights_path: str,
    full_output: bool = False
) -> Optional[Union[float, Dict]]:
    """
    Predict crystal density from SMILES string using trained weights.
    
    For single molecules: uses calibrated bond overlap dictionaries (MW-split).
    For co-crystals (SMILES with '.'): uses mass-weighted average of component densities.
    
    Args:
        smiles: SMILES string (may contain '.' for co-crystals)
        weights_path: Path to trained weights .pkl file
        full_output: If True, returns dictionary with all details.
        
    Returns:
        Density in g/cm³ (float) or dict if full_output=True.
    """
    # Load weights
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    
    # Extract strategy and defaults
    metadata = weights.get('metadata', {})
    strategy = metadata.get('strategy', 'single')
    default_overlap = weights.get('default_overlap', DEFAULT_OVERLAP)
    
    # Co-crystal: weighted average of components
    if '.' in smiles:
        components = smiles.split('.')
        comp_results = []
        
        for comp in components:
            rho = predict_density_single(comp, weights, strategy, default_overlap, full_output=False)
            mw = get_molecular_weight(comp)
            if rho is None or mw is None:
                return None
            comp_results.append({'smiles': comp, 'density': rho, 'mw': mw})
        
        total_mw = sum(c['mw'] for c in comp_results)
        weighted_density = sum(c['mw'] * c['density'] for c in comp_results) / total_mw
        
        if not full_output:
            return weighted_density
        
        return {
            'density': weighted_density,
            'total_mw': total_mw,
            'components': comp_results,
            'weight_fractions': [c['mw'] / total_mw for c in comp_results],
            'n_components': len(components),
            'method': 'ratio',
            'strategy': strategy,
        }
    
    # Single molecule
    return predict_density_single(smiles, weights, strategy, default_overlap, full_output)


def predict_density_single(
    smiles: str,
    weights: dict,
    strategy: str,
    default_overlap: float,
    full_output: bool = False
) -> Optional[Union[float, Dict]]:
    """
    Predict density for a single molecule (internal function).
    """
    mw = get_molecular_weight(smiles)
    if mw is None:
        return None
    
    # Get appropriate dictionary
    overlaps_dict, class_name, fallback_used = get_overlaps_for_molecule(
        smiles, mw, weights, strategy
    )
    
    density = predict_single_molecule(smiles, overlaps_dict, default_overlap)
    
    if density is None:
        return None
    
    if not full_output:
        return density
    
    # Calculate volume for full output
    V_magic = magic_volume(smiles)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    total_overlap = 0.0
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetSymbol()
        a2 = bond.GetEndAtom().GetSymbol()
        bt = bond.GetBondType()
        order = 1 if bt == Chem.rdchem.BondType.SINGLE else 2 if bt == Chem.rdchem.BondType.DOUBLE else 3
        key = make_key(a1, a2, order)
        total_overlap += overlaps_dict.get(key, default_overlap)
    
    V_corrected = V_magic - total_overlap
    if V_corrected <= 0:
        V_corrected = V_magic * 0.5
    
    return {
        'density': density,
        'mw': mw,
        'V_magic': V_magic,
        'V_corrected': V_corrected,
        'total_overlap': total_overlap,
        'n_bonds': mol.GetNumBonds(),
        'method': 'dictionary',
        'strategy': strategy,
        'class_used': class_name,
        'fallback_used': fallback_used,
        'mw_regime': 'small' if mw < MW_SMALL_MAX else 'medium' if mw < MW_MEDIUM_MAX else 'large'
    }


def predict_density_batch(
    smiles_list: List[str],
    weights_path: str,
    full_output: bool = False,
    verbose: bool = False,
    n_cores: int = DEFAULT_CORES,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> List[Optional[Union[float, Dict]]]:
    """
    Predict densities for multiple SMILES strings.
    """
    # If full_output requested, fallback to single-core with full logic
    if full_output:
        if verbose:
            print("Note: full_output=True with multiprocessing not fully supported. Using single-core.")
        results = []
        iterator = tqdm(smiles_list, desc="Predicting", disable=not verbose)
        for smiles in iterator:
            result = predict_density(smiles, weights_path, full_output=True)
            results.append(result)
        return results
    
    # 👇 LOAD WEIGHTS ONCE
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    
    # Extract dictionaries for fast path
    metadata = weights.get('metadata', {})
    strategy = metadata.get('strategy', 'single')
    default_overlap = weights.get('default_overlap', DEFAULT_OVERLAP)
    
    # Build dictionaries
    if strategy == 'single':
        dict_small = weights.get('global', {})
        dict_medium = weights.get('global', {})
        dict_large = weights.get('global', {})
        dict_coco = weights.get('global', {})
    elif strategy == 'two_class':
        merged_dict = {}
        if 'small' in weights:
            merged_dict.update(weights['small'])
        if 'medium' in weights:
            merged_dict.update(weights['medium'])
        dict_small = merged_dict
        dict_medium = merged_dict
        dict_large = weights.get('large', {})
        dict_coco = weights.get('global', {}) if 'global' in weights else merged_dict
    elif strategy == 'three_class':
        dict_small = weights.get('small', {})
        dict_medium = weights.get('medium', {})
        dict_large = weights.get('large', {})
        dict_coco = weights.get('global', {}) if 'global' in weights else dict_medium
    else:
        dict_small = weights.get('small', {})
        dict_medium = weights.get('medium', {})
        dict_large = weights.get('large', {})
        dict_coco = weights.get('global', {}) if 'global' in weights else dict_medium
    
    # Single-core fast path (no multiprocessing)
    if n_cores <= 1:
        results = []
        iterator = tqdm(smiles_list, desc="Predicting", disable=not verbose)
        for smiles in iterator:
            result = _predict_single_with_dicts_fast(
                smiles, dict_small, dict_medium, dict_large, dict_coco, default_overlap
            )
            results.append(result)
        return results
    
    # Multiprocessing path (same as before)
    if verbose:
        print(f"Using {n_cores} CPU cores")
        print(f"  Chunk size: {chunk_size}, chunks: {len(smiles_list) // chunk_size + 1}")
    
    chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    predict_func = partial(
        _predict_batch_chunk_fast,
        dict_small=dict_small,
        dict_medium=dict_medium,
        dict_large=dict_large,
        dict_coco=dict_coco,
        default_overlap=default_overlap
    )
    
    results = [None] * len(smiles_list)
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        future_to_chunk = {
            executor.submit(predict_func, chunk): idx 
            for idx, chunk in enumerate(chunks)
        }
        
        if verbose:
            for future in tqdm(
                as_completed(future_to_chunk),
                total=len(chunks),
                desc="Processing chunks"
            ):
                chunk_idx = future_to_chunk[future]
                chunk_results = future.result()
                start_idx = chunk_idx * chunk_size
                for i, pred in enumerate(chunk_results):
                    if start_idx + i < len(results):
                        results[start_idx + i] = pred
        else:
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                chunk_results = future.result()
                start_idx = chunk_idx * chunk_size
                for i, pred in enumerate(chunk_results):
                    if start_idx + i < len(results):
                        results[start_idx + i] = pred
    
    return results


def get_dictionary_info(weights_path: str) -> Dict[str, Union[int, str]]:
    """
    Get information about loaded weights file.
    
    Args:
        weights_path: Path to trained weights .pkl file
        
    Returns:
        Dictionary with metadata and dictionary sizes.
    """
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    
    metadata = weights.get('metadata', {})
    info = {
        'strategy': metadata.get('strategy', 'unknown'),
        'n_molecules': metadata.get('n_molecules', 0),
        'source_data': metadata.get('source_data', 'unknown'),
        'date_trained': metadata.get('date_trained', 'unknown'),
        'default_overlap': weights.get('default_overlap', DEFAULT_OVERLAP),
    }
    
    # Dictionary sizes
    for key in ['small', 'medium', 'large', 'global']:
        if key in weights:
            info[f'n_{key}'] = len(weights[key])
        else:
            info[f'n_{key}'] = 0
    
    return info


# ============================================================================
# LEGACY SUPPORT (for backward compatibility)
# ============================================================================

def predict_density_legacy(smiles: str, weights_path: str, full_output: bool = False) -> Optional[Union[float, Dict]]:
    """
    Legacy predictor using the old quadratic/conformer model.
    Kept for backward compatibility.
    """
    return predict_density(smiles, weights_path, full_output)


# ============================================================================
# COMPATIBILITY ALIASES
# ============================================================================

# Alias for backward compatibility
predict_density_fast = predict_density
predict_single = predict_single_molecule


# ============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# ============================================================================

def predict_density_with_k(
    smiles: str,
    k: float,
    weights_path: str,
    full_output: bool = False
) -> Optional[Union[float, Dict]]:
    """
    Legacy function: Predict density using a fixed packing coefficient.
    
    Note: This is a wrapper for backward compatibility.
    For best accuracy, use predict_density() without specifying k.
    
    Args:
        smiles: SMILES string (may contain '.' for co-crystals)
        k: Packing coefficient (typically 0.60-0.77)
        weights_path: Path to trained weights .pkl file
        full_output: If True, returns dictionary with details
        
    Returns:
        Density in g/cm³ (float) or dict if full_output=True.
    """
    # Get base density
    rho = predict_density(smiles, weights_path, full_output=False)
    if rho is None:
        return None
    
    # Estimate effective k from the model
    mw = get_molecular_weight(smiles)
    if mw is None:
        return None
    
    V_magic = magic_volume(smiles)
    if V_magic is None:
        return None
    
    # Calculate effective k
    effective_k = rho * V_magic / (mw * G_FACTOR)
    
    # Scale density to requested k
    scaled_rho = rho * (k / effective_k) if effective_k > 0 else rho
    
    if not full_output:
        return scaled_rho
    
    return {
        'density': scaled_rho,
        'mw': mw,
        'k_requested': k,
        'k_effective': effective_k,
        'method': 'legacy_k_wrapper'
    }


def predict_density_experimental(
    smiles: str,
    weights_path: str,
    a: Optional[float] = None,
    b: Optional[float] = None,
    c: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    no_weights: bool = False,
    full_output: bool = True
) -> Optional[Union[float, Dict]]:
    """
    Legacy function: Experimental quadratic model with custom weights.
    
    Note: This is a wrapper for backward compatibility.
    For best accuracy and speed, use predict_density() instead.
    
    Args:
        smiles: SMILES string
        weights_path: Path to trained weights .pkl file
        a, b, c: Quadratic coefficients (k = a·MW² + b·MW + c)
        weights: CHON weights for bump correction
        no_weights: If True, bump = 0
        full_output: If True, returns dictionary
        
    Returns:
        Density in g/cm³ (float) or dict if full_output=True.
    """
    # For co-crystals, use ratio method
    if '.' in smiles:
        components = smiles.split('.')
        comp_results = []
        
        for comp in components:
            result = predict_density_experimental(
                comp, weights_path, a=a, b=b, c=c, 
                weights=weights, no_weights=no_weights, full_output=True
            )
            if result is None:
                return None
            comp_results.append(result)
        
        total_mw = sum(r['mw'] for r in comp_results)
        weighted_density = sum(r['mw'] * r['density'] for r in comp_results) / total_mw
        
        if not full_output:
            return weighted_density
        
        return {
            'density': weighted_density,
            'total_mw': total_mw,
            'components': comp_results,
            'method': 'legacy_experimental_ratio'
        }
    
    # Single molecule: use standard predictor
    rho = predict_density(smiles, weights_path, full_output=False)
    if rho is None:
        return None
    
    if not full_output:
        return rho
    
    mw = get_molecular_weight(smiles)
    return {
        'density': rho,
        'mw': mw,
        'method': 'legacy_experimental_wrapper',
        'note': 'Using fast dictionary model (legacy parameters ignored)'
    }


# ============================================================================
# TRACEABILITY FUNCTIONS
# ============================================================================

def explain_prediction(
    smiles: str,
    weights_path: str,
    property: str = "density"
) -> None:
    """
    Print a readable trace of how the density prediction was calculated.
    
    Args:
        smiles: SMILES string
        weights_path: Path to trained weights .pkl file
        property: Only "density" is supported (HOF removed in v3.0.0)
    
    Prints a table showing each calculation step.
    """
    print("\n" + "=" * 70)
    print(f"🔮 MAGIC VOLUME TRACE: Density prediction for {smiles}")
    print("=" * 70)
    _trace_density(smiles, weights_path)


def _trace_density(smiles: str, weights_path: str) -> None:
    """Print density calculation trace."""
    # Load weights
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    
    metadata = weights.get('metadata', {})
    strategy = metadata.get('strategy', 'single')
    default_overlap = weights.get('default_overlap', DEFAULT_OVERLAP)
    
    mw = get_molecular_weight(smiles)
    if mw is None:
        print("  ❌ Could not calculate molecular weight")
        return
    
    V_magic = magic_volume(smiles)
    if V_magic is None:
        print("  ❌ Could not calculate magic volume")
        return
    
    overlaps_dict, class_name, fallback_used = get_overlaps_for_molecule(
        smiles, mw, weights, strategy
    )
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("  ❌ Could not parse SMILES")
        return
    mol = Chem.AddHs(mol)
    
    # Collect bond overlaps
    bond_list = []
    total_overlap = 0.0
    
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetSymbol()
        a2 = bond.GetEndAtom().GetSymbol()
        if a1 > a2:
            a1, a2 = a2, a1
        bt = bond.GetBondType()
        if bt == Chem.rdchem.BondType.SINGLE:
            order = 1
        elif bt == Chem.rdchem.BondType.DOUBLE:
            order = 2
        elif bt == Chem.rdchem.BondType.TRIPLE:
            order = 3
        else:
            order = 3
        
        key = make_key(a1, a2, order)
        overlap = overlaps_dict.get(key, default_overlap)
        total_overlap += overlap
        bond_list.append((f"{a1}-{a2}", order, overlap))
    
    V_corrected = V_magic - total_overlap
    if V_corrected <= 0:
        V_corrected = V_magic * 0.5
    density = mw / V_corrected
    
    # Print table
    print("\n📦 DENSITY PREDICTION")
    print("-" * 50)
    print(f"  Strategy:                     {strategy.upper()}")
    print(f"  Class used:                   {class_name}" + (" (fallback)" if fallback_used else ""))
    print(f"  Molecular weight:              {mw:.2f} g/mol")
    print(f"  Magic volume:                  {V_magic:.2f} Å³")
    print(f"  Total bond overlap:            {total_overlap:.4f} Å³")
    print(f"  Corrected volume:              {V_corrected:.2f} Å³")
    print(f"\n  ✨ Predicted density:          {density:.4f} g/cm³")
    print(f"\n  Formula: Magic volume = π × φ × Σ(mass^(1/3)) = {V_magic:.2f} Å³")
    
    print(f"\n  Bond overlaps (subtracted from magic volume):")
    for i, (bond, order, overlap) in enumerate(bond_list, 1):
        print(f"    {i:2d}. {bond} (order {order}): {overlap:.4f} Å³")


def print_dictionary_weights(weights_path: str):
    """
    Print all bond overlap weights from the density dictionaries.
    Shows the compression value (in Å³) for each bond type.
    """
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    
    metadata = weights.get('metadata', {})
    strategy = metadata.get('strategy', 'unknown')
    
    print("\n" + "=" * 70)
    print(f"DENSITY DICTIONARY WEIGHTS (Bond Overlaps in Å³)")
    print(f"Strategy: {strategy.upper()}")
    print("=" * 70)
    
    # Determine which dicts exist
    dict_names = []
    if strategy == 'single' and 'global' in weights:
        dict_names = [('Global', weights['global'])]
    elif strategy == 'two_class':
        if 'medium' in weights:
            dict_names.append(('Medium (<400)', weights['medium']))
        if 'large' in weights:
            dict_names.append(('Large (>=400)', weights['large']))
        if 'small' in weights:
            dict_names.append(('Small (<180)', weights['small']))
    elif strategy == 'three_class':
        if 'small' in weights:
            dict_names.append(('Small (<180)', weights['small']))
        if 'medium' in weights:
            dict_names.append(('Medium (180-400)', weights['medium']))
        if 'large' in weights:
            dict_names.append(('Large (>400)', weights['large']))
    else:
        # Fallback: show all
        for key in ['small', 'medium', 'large', 'global']:
            if key in weights:
                dict_names.append((key.capitalize(), weights[key]))
    
    for name, d in dict_names:
        if d:
            print(f"\n{name}: ({len(d)} bond types)")
            print("-" * 40)
            for key, value in sorted(d.items(), key=lambda x: x[1], reverse=True):
                if len(key) == 3:
                    a1, a2, order = key
                    print(f"  {a1}-{a2} (order {order}): {value:.4f} Å³")
                else:
                    print(f"  {key}: {value:.4f} Å³")
        else:
            print(f"\n{name}: Empty")