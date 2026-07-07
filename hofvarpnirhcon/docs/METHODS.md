## 1 Introducation ##

Crystal density is a fundamental descriptor in crystal engineering, influencing packing efficiency, polymorph stability, solubility, and mechanical behavior. Accurate density prediction is therefore essential for pharmaceutical development, energetic materials research, and high-throughput screening of functional organic crystals. Despite its importance, reliable prediction remains challenging because it depends on both molecular structure and intermolecular packing.

Traditional approaches fall into three broad categories. First, empirical additivity methods and van der Waals volume-based estimates are computationally fast but often suffer from systematic errors because they do not account for bond-specific volume contractions or packing variations. Second, machine learning models — including random forests, gradient-boosted trees, and kernel methods — achieve higher accuracy but require large curated datasets and periodic retraining, and they often lack physical interpretability. Third, graph neural networks (GNNs) and deep learning models have shown impressive performance, yet they rely on tensor operations, often require GPU acceleration, and are sensitive to training data biases.

At the highest level of accuracy, density functional theory (DFT) and periodic crystal-structure prediction (CSP) workflows can predict crystal densities from first principles. However, these methods are computationally expensive, often requiring hours to days per molecule, and are impractical for high-throughput screening of thousands or millions of candidates.

A further limitation of many existing methods is their treatment of co-crystals. Most predictors are designed for single-component crystals and cannot seamlessly handle multi-component systems without extensive retraining or specialized descriptors.

Here we introduce HófvarpnirHCON, a fast, dictionary-based crystal density predictor that achieves accuracy comparable to state-of-the-art machine learning models (MAE ≈ 0.0302 g cm⁻³ for HCON molecules). The bond overlap dictionaries are trained once from experimental data using Non-Negative Least Squares (NNLS), a lightweight linear optimization that converges rapidly and requires no GPU, no neural networks, and no quantum mechanical calculations. After training, predictions reduce to dictionary lookups and arithmetic, enabling inference at speeds of approximately 1,800 predictions per second on a standard CPU core.

In this paper, we describe the model, its implementation, and its validation on publicly available experimental datasets. The code is open-source under the BSD-3 license and will be made available upon publication.

## 2 Methods ##

## 2.1 Overview of the Prediction Framework

HófvarpnirHCON predicts crystal density from a SMILES string using a two-step procedure. First, the Leonardus Volume is calculated from the sum of cube roots of atomic masses. Second, bond-specific overlap corrections — precomputed and stored in molecular-weight-stratified dictionaries — are subtracted from this Leonardus volume to obtain the corrected molecular volume. Density is then computed as molecular weight divided by the corrected volume. For co-crystals (SMILES strings containing a dot character), the predictor recursively computes densities for each component and returns the mass-weighted average.

## 2.2 Leonardus Volume ##

The starting point is the Leonardus Volume, an idealized molecular volume calculated directly from atomic masses:

\[
V_L = \pi \cdot \phi \cdot \sum_{i} m_i^{1/3}
\]

where:
- \(\phi = \frac{1+\sqrt{5}}{2}\) (the golden ratio)
- \(m_i\) is the atomic mass of atom \(i\)
- The sum runs over all atoms in the molecule (including hydrogens)

This volume represents an upper-bound estimate of molecular packing before bond-specific overlap corrections are applied.

## 2.3 Bond Overlap Dictionaries ##

The core of the predictor is a set of dictionaries that map bond types to volume overlap values (in Å³). Each dictionary entry corresponds to a tuple (atom1, atom2, bond_order), with atom symbols alphabetically sorted. For example, a carbon–oxygen single bond is stored under the key ('C', 'O', 1). The overlap value represents the volume reduction (in Å³) associated with the formation of that bond, relative to the sum of isolated atomic volumes.

These overlap values were derived by fitting to experimental crystal density data. For each bond type, the overlap parameter was optimized to minimize prediction error across a training set of pure organic crystals and then fixed in the final dictionaries. No further training or optimization occurs at prediction time.


## 2.4 Molecular Weight Stratification ##

Because bond overlap effects scale with molecular size, a single dictionary produces systematic errors across the molecular weight spectrum. HófvarpnirHCON therefore uses three separate dictionaries for pure molecules, stratified by molecular weight:

- Small molecules: MW < 180 g/mol
- Medium molecules: 180 ≤ MW < 400 g/mol  
- Large molecules: MW ≥ 400 g/mol

For each input molecule, the molecular weight is computed from the SMILES string (with explicit hydrogens), and the appropriate dictionary is selected automatically.

## 2.5 Co-crystal Prediction ##

For co-crystals, the input SMILES string contains a dot (e.g., "CCO.O=C(O)C" for an ethanol–oxalic acid co-crystal). HófvarpnirHCON splits the string on the dot and processes each component independently using the pure-molecule pipeline. The overall density is then computed as the mass-weighted average:

\[
\rho_{\text{cocrystal}} = \frac{\sum_{j} m_j \rho_j}{\sum_{j} m_j}
\]

where \(m_j\) and \(\rho_j\) are the molecular weight and predicted density of component \(j\), respectively. No specialized co-crystal dictionaries or parameters are required; the mass-weighted average emerges naturally from volume additivity.

For datasets containing a large number of co-crystals, improved accuracy can be achieved by training separate dictionaries on co-crystal data only. For datasets with only a few co-crystals, the pure-trained dictionaries provide reliable predictions via mass-weighted averaging, and no special treatment is required.


## 2.6 Prediction Algorithm ##

The complete prediction procedure for a pure molecule is as follows:

1. Parse the SMILES string using RDKit and add explicit hydrogens.

2. Compute the molecular weight \( M \) as the sum of atomic masses.

3. Calculate the Leonardus Volume \( V_L = \pi \cdot \phi \cdot \sum m_i^{1/3} \).

4. Select the appropriate overlap dictionary based on \( M \).

5. Iterate over all bonds in the molecule:
   - Determine the atom types and bond order (1, 2, or 3)
   - Look up the overlap value \( \delta \) in the dictionary
   - Sum to obtain total overlap \( \Delta V = \sum \delta_{\text{bond}} \)

6. Compute corrected volume \( V_{\text{corr}} = V_L - \Delta V \).

7. If \( V_{\text{corr}} \le 0 \), set \( V_{\text{corr}} = 0.5 \times V_L \).

8. Return density \( \rho = M / V_{\text{corr}} \) (in g/cm³).

For co-crystals, the algorithm recurses on each component and returns the mass-weighted density.

## 2.7 Implementation Details

HófvarpnirHCON is implemented in Python 3.8+ and depends only on NumPy (for numerical constants) and RDKit (for SMILES parsing and molecular graph traversal). No machine learning libraries are required for prediction. Dictionaries are stored as Python pickle files and loaded lazily on first use.

## 2.8 Data and Code Availability ##

Experimental crystal density data are available from the Davis and Mathieu datasets, which are publicly available as Supporting Information with their respective publications.