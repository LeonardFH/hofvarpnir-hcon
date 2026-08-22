
markdown
<p align="center">
  <img src="images/hofvarpnir_logo.png" alt="HófvarpnirHCON Logo" width="300">
</p>

# Reproducing the ChemRxiv Paper Results

This folder contains all scripts used to generate the results reported in the ChemRxiv preprint:

> Haasbroek, L. F. (2026). *HófvarpnirHCON: An Interpretable Composition-Derived Volume and Bond-Overlap Dictionary Model for Fast Crystal Density Prediction from SMILES.* ChemRxiv. DOI: [10.XXXX/chemrxiv-2026-XXXXX](https://doi.org/10.XXXX/chemrxiv-2026-XXXXX)

---

## Data Sources

The training and evaluation data used in the paper may be obtained from the following publicly available sources:

- **Taniguchi (2025)**: Taniguchi, T.; Fukasawa, R. Crystal Structure Prediction of Organic Molecules by Machine Learning-Based Lattice Sampling and Structure Relaxation. *Digital Discovery* **2025**, *4*, 3270–3281. DOI: [10.1039/d5dd00304k](https://doi.org/10.1039/d5dd00304k)
  - **Dataset:** [SPaDe-CSP GitHub](https://github.com/takuyhaa/SPaDe-CSP) | Zenodo DOI: [10.5281/zenodo.17214315](https://doi.org/10.5281/zenodo.17214315)

- **He (2026)**: He, Y.-J. et al. Transfer learning-enabled density prediction model for energetic molecule screening. *Energetic Materials Frontiers* **2026**, *7* (2), 152–160. DOI: [10.1016/j.enmf.2025.11.010](https://doi.org/10.1016/j.enmf.2025.11.010)
  - **Dataset:** [TransfLearn GitHub](https://github.com/caepliujian/TransfLearn)

- **Davis (2024)**: Davis, J. V. et al. Machine Learning Models for High Explosive Crystal Density and Performance. *Chem. Mater.* **2024**, *36*, 11109–11118. DOI: [10.1021/acs.chemmater.4c01978](https://doi.org/10.1021/acs.chemmater.4c01978)

- **Jin (2023)**: Jin, J.-X. et al. Force field-inspired transformer network assisted crystal density prediction for energetic materials. *J. Cheminform.* **2023**, *15*, 65. DOI: [10.1186/s13321-023-00736-6](https://doi.org/10.1186/s13321-023-00736-6)
  - **Dataset:** [FFiTrNet GitHub](https://github.com/jjx-2000/FFiTrNet)

- **Taylor (2025)**: Taylor, C. R. et al. Predictive Crystallography at Scale: Mapping, Validating, and Learning from 1,000 Crystal Energy Landscapes. *Faraday Discuss.* **2025**, *256*, 434–458. DOI: [10.1039/D4FD00105B](https://doi.org/10.1039/D4FD00105B)
  - **Dataset:** University of Southampton Repository. DOI: [10.5258/SOTON/D3094](https://doi.org/10.5258/SOTON/D3094)
  
- **Mathieu (2017)**: Mathieu, D. Sensitivity of Energetic Materials: Theoretical Relationships to Detonation Performance and Molecular Structure. *Ind. Eng. Chem. Res.* **2017**, *56*, 8191–8201. DOI: [10.1021/acs.iecr.7b02021](https://doi.org/10.1021/acs.iecr.7b02021)

These datasets are available as Supporting Information with their respective papers or via the linked public repositories.

---

## Python Programs

| Script | What It Does | Paper Output |
|--------|--------------|--------------|
| `leonardus_volume_discovery.py` | Stage 1: Tests raw mass (V = Σm) and unscaled cubic-root (V = Σm^(1/3)) as volume proxies. | **Table 3** |
| `leonardus_scaling_optimization.py` | Stage 2: Finds the optimal global scaling factor C₀ for each dataset. | **Table 4**, **Figure 1** |
| `tenfold_cross_validation.py` | 10‑fold cross‑validation on all six datasets with automatic MW stratification. | **Table 8**, **Table 9** |
| `molecular_weight_stratification.py` | Compares single vs automatic molecular weight class strategies. | **Table 7** |
| `chemical_family_stratification.py` | Evaluates performance across CHON, S, F, Cl, Br, P, I families. | **Table 10**, **Table 11** |
| `half_split_transfer.py` | Tests dictionary transferability across independent dataset splits. | **Table 12**, **Table 13** |
| `synthetic_closure.py` | Checks whether the NNLS dictionary can be regenerated from its own predictions. | **Table 14**, **Table 15**, **Table 16** |
| `split_half_stability.py` | Compares coefficients from dictionaries trained on independent splits. | **Table 17**, **Figure 4** |
| `davis_taniguchi_comparison.py` | Overlaps 2,699 identical molecules between Davis and Taniguchi datasets. | **Table 18**, **Table 19**, **Figure 5** |
| `four_way_intersection.py` | 1,007 molecules common to Taniguchi, He, Davis, Jin; isolates dataset curation bias. | **Table 20**, **Table 21** |
| `polymorph_comparison.py` | Compares predicted density against experimental polymorphs (ROY, carbamazepine, aspirin, paracetamol, sulfathiazole, ritonavir). | **Figure 6** |
| `inference_benchmark.py` | Measures throughput vs molecular weight on a 9‑core laptop. | **Table 22** |
| `benchmark_comparison.py` | Compares HófvarpnirHCON against published models (Nguyen, Jin, Davis, He, Taniguchi). | **Table 23** |
| `co_crystal_demo.py` | (Optional) Demonstrates co‑crystal prediction using mass‑weighted averaging. | Not in paper |

## Quick Start

# Install dependencies
pip install numpy pandas rdkit scikit-learn tqdm matplotlib
