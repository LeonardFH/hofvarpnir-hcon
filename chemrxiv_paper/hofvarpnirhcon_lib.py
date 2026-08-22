"""
HÓFVARPNIRHCON — Core Library
Molecular property calculation engine for ZINC22 processing
"""

import numpy as np
from rdkit import Chem

# ============================================================
# MODEL STRUCTURAL CONSTANTS
# ============================================================
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
LEONARDUS_SCALE = (PI ** 2) / PHI
MW_SMALL_MAX = 180
MW_MEDIUM_MAX = 400

# ============================================================
# ATOMIC MASSES
# ============================================================
ATOM_MASSES = {
    'H': 1.008, 'C': 12.011, 'O': 15.999, 'N': 14.007, 'S': 32.065, 'P': 30.974,
    'F': 18.998, 'Cl': 35.450, 'Br': 79.904, 'I': 126.904, 'Li': 6.941, 
    'Na': 22.990, 'K': 39.098, 'Mg': 24.305, 'Ca': 40.078, 'Fe': 55.845, 
    'Zn': 65.380, 'Cu': 63.546, 'Mn': 54.938, 'Al': 26.982, 'Si': 28.086, 
    'Se': 78.960, 'As': 74.922,
}

# ============================================================
# CORE FUNCTIONS
# ============================================================
def make_key(atom1: str, atom2: str, bond_order: int) -> tuple:
    """
    Create alphabetically sorted bond key matching your original model.
    
    Args:
        atom1: First atom symbol
        atom2: Second atom symbol
        bond_order: Bond order (1, 2, or 3)
    
    Returns:
        Tuple of (atom1, atom2, bond_order) sorted alphabetically
    """
    if atom1 <= atom2:
        return (atom1, atom2, bond_order)
    return (atom2, atom1, bond_order)


def calculate_molecule_properties_direct(
    smiles: str, 
    dict_small: dict, 
    dict_medium: dict, 
    dict_large: dict, 
    default_overlap: float
):
    """
    Executes your exact physics loop sequence directly without caching layers.
    
    Args:
        smiles: SMILES string of the molecule
        dict_small: Overlap dictionary for small molecules (MW < 180)
        dict_medium: Overlap dictionary for medium molecules (180 < MW < 400)
        dict_large: Overlap dictionary for large molecules (MW > 400)
        default_overlap: Default overlap value if bond key not found
    
    Returns:
        Tuple of (element_string, predicted_density) or (None, None) on failure
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None, None
        
    # 1. Extract heavy atom element string signature
    elem_str = "".join(sorted({atom.GetSymbol() for atom in mol.GetAtoms()}))
    
    # 2. Fully hydrate graph connectivity states
    mol = Chem.AddHs(mol)
    
    # 3. Calculate Molecular Weight
    mw = 0.0
    for atom in mol.GetAtoms():
        mw += ATOM_MASSES.get(atom.GetSymbol(), 0.0)
    if mw <= 0: 
        return None, None
    
    # 4. Calculate Leonardus Volume
    total_cuberoot = 0.0
    for atom in mol.GetAtoms():
        mass = ATOM_MASSES.get(atom.GetSymbol(), 0.0)
        total_cuberoot += mass ** (1/3)
    V_L = LEONARDUS_SCALE * total_cuberoot
    if V_L <= 0: 
        return None, None
    
    # 5. Route directly to target dictionary stratification class based on calculated MW
    if mw < MW_SMALL_MAX:
        overlaps_dict = dict_small
    elif mw < MW_MEDIUM_MAX:
        overlaps_dict = dict_medium
    else:
        overlaps_dict = dict_large
        
    if not overlaps_dict:
        return None, None
        
    # 6. Accumulate bond corrections using your original make_key loop structure
    total_overlap = 0.0
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetSymbol()
        a2 = bond.GetEndAtom().GetSymbol()
        bt = bond.GetBondType()
        
        if bt == Chem.rdchem.BondType.SINGLE:
            order = 1
        elif bt == Chem.rdchem.BondType.DOUBLE:
            order = 2
        else:
            order = 3
            
        key = make_key(a1, a2, order)
        total_overlap += overlaps_dict.get(key, default_overlap)
        
    V_corrected = V_L - total_overlap
    if V_corrected <= 0:
        V_corrected = V_L * 0.5
        
    return elem_str, (mw / V_corrected)


def element_string(mol):
    """
    Extract element string from an RDKit molecule.
    
    Args:
        mol: RDKit Mol object
    
    Returns:
        Sorted string of element symbols present in the molecule
    """
    return "".join(sorted({atom.GetSymbol() for atom in mol.GetAtoms()}))


def parse_smiles_line(line):
    """
    Parse a line from a ZINC file into SMILES and ZINC ID.
    
    Args:
        line: Line from a ZINC file
    
    Returns:
        Tuple of (SMILES, ZINC_ID) or (None, None) on failure
    """
    toks = line.split()
    if len(toks) >= 2:
        if toks[1].upper().startswith("ZINC"):
            return toks[0], toks[1]
        if toks[0].upper().startswith("ZINC"):
            return toks[1], toks[0]
    return None, None


def safe_calculate(smiles, dict_small, dict_medium, dict_large, default_overlap):
    """
    Safe wrapper for calculate_molecule_properties_direct that catches all exceptions.
    
    Returns:
        Tuple of (elem_str, pred_density) or (None, None) on any failure
    """
    try:
        return calculate_molecule_properties_direct(
            smiles, dict_small, dict_medium, dict_large, default_overlap
        )
    except Exception:
        return None, None


# ============================================================
# TEST FUNCTION (OPTIONAL)
# ============================================================
def test_library():
    """Quick test to verify the library is working."""
    print("🧪 Testing HÓFVARPNIRHCON library...")
    
    # Test constants
    print(f"  ✓ PHI = {PHI:.6f}")
    print(f"  ✓ PI = {PI:.6f}")
    print(f"  ✓ LEONARDUS_SCALE = {LEONARDUS_SCALE:.6f}")
    print(f"  ✓ MW_SMALL_MAX = {MW_SMALL_MAX}")
    print(f"  ✓ MW_MEDIUM_MAX = {MW_MEDIUM_MAX}")
    
    # Test atom masses
    print(f"  ✓ Found {len(ATOM_MASSES)} elements in mass table")
    
    # Test make_key
    key1 = make_key("C", "O", 2)
    key2 = make_key("O", "C", 2)
    assert key1 == key2, "make_key failed: keys should be sorted"
    print(f"  ✓ make_key works: {key1}")
    
    # Test with a simple molecule
    test_smiles = "CC(=O)O"  # Acetic acid
    dict_small = {}
    dict_medium = {}
    dict_large = {}
    default_overlap = 0.3
    
    result = calculate_molecule_properties_direct(
        test_smiles, dict_small, dict_medium, dict_large, default_overlap
    )
    
    if result is not None:
        elem_str, density = result
        print(f"  ✓ calculate_molecule_properties_direct works!")
        print(f"    SMILES: {test_smiles}")
        print(f"    Elements: {elem_str}")
        print(f"    Density: {density:.4f}")
    else:
        print(f"  ⚠️  calculate_molecule_properties_direct returned None for {test_smiles}")
    
    print("✅ Library test complete!")


if __name__ == "__main__":
    test_library()