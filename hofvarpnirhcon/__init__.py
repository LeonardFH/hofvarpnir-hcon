"""
HófvarpnirHCON : Fast Dictionary-Based Crystal Density Prediction

A fast, physics-based predictor of crystal density from SMILES strings,
optimized for organic molecules containing C, H, N, O.

For detailed methods documentation, run:

    from hofvarpnirhcon import show_docs
    show_docs()

Functions:
    predict_density: Fast dictionary model for crystal density
    predict_density_batch: Batch prediction for multiple molecules
    get_dictionary_info: Information about loaded density dictionaries
    get_molecular_weight: Calculate molecular weight from SMILES
    train_density: Train density dictionaries from CSV data

Co-crystal support (SMILES with '.'):
    - For co-crystals: mass-weighted average of pure component densities
    - Handles any number of components recursively
"""

from .predictor import (
    predict_density,
    predict_density_batch,
    get_molecular_weight,
    get_dictionary_info,
    print_dictionary_weights,
    # Legacy wrappers (for backward compatibility)
    predict_density_with_k,
    predict_density_experimental,
)
from .train_density import train_density  
from .version import __version__

# ============================================================================
# DOCUMENTATION HELPER
# ============================================================================

def show_docs():
    """
    Print the full methods documentation for HófvarpnirHCON.
    
    This prints the contents of docs/METHODS.md to the console.
    """
    import os
    
    # Try to find the docs file
    doc_path = os.path.join(os.path.dirname(__file__), 'docs', 'METHODS.md')
    
    # If not found, try looking in the parent directory (source install)
    if not os.path.exists(doc_path):
        doc_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'METHODS.md')
    
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:  # 👈 FIX: added encoding
            print("\n" + "=" * 70)
            print("HÓFVARPNIRHCON - METHODS DOCUMENTATION")
            print("=" * 70 + "\n")
            print(f.read())
            print("\n" + "=" * 70)
    except FileNotFoundError:
        print("\n❌ Documentation file not found.")
        print("Please view it online at: https://github.com/yourusername/hofvarpnir-hcon/docs/METHODS.md")

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'predict_density',
    'predict_density_batch',
    'get_dictionary_info',
    'get_molecular_weight',
    'print_dictionary_weights',
    'predict_density_with_k',
    'predict_density_experimental',
    'train_density', 
    'show_docs',  
    '__version__'
]