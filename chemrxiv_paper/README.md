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

### Discovery & Baseline
| Script | What It Does | Paper Output |
|--------|--------------|--------------|
| `leonardus_volume_discovery.py` | Stage 1: Tests raw mass (V = Σm) and unscaled cubic-root (V = Σm^(1/3)) as volume proxies. | **Table 3** |
| `leonardus_scaling_optimization.py` | Stage 2: Finds the optimal global scaling factor C₀ for each dataset. | **Table 4**, **Figure 1** |

### Convergence & Stability
| Script | What It Does | Paper Output |
|--------|--------------|--------------|
| `learning_curve_convergence.py` | Trains on increasing dataset sizes to measure dictionary convergence and stability. | **Figure 2** |
| `learning_curve_convergence_plot.py` | Generates the convergence figure from summary data. | **Figure 2** |
| `slice_stability.py` | Trains on 2,500‑molecule slices and predicts remaining molecules only. Tests generalisation to unseen data. | **Figure 3**, **Table 6** |
| `slice_stability_combined.py` | Generates combined figure: (a) scatter of MAE per slice, (b) violin distribution. | **Figure 3** |
| `slice_stability_statistics.py` | Calculates summary statistics (mean, median, std, CV, IQR, CI) for Table 6. | **Table 6** |

### Molecular Weight & Chemical Family Stratification
| Script | What It Does | Paper Output |
|--------|--------------|--------------|
| `single_vs_auto.py` | Compares single global dictionary against automatic molecular weight class stratification. | **Table 7** |
| `chemical_family_stratification.py` | Evaluates performance across CHON, S, F, Cl, Br, P, I families. Runs single vs auto MW strategy for each family. | **Table 10**, **Table 11** |

### Cross-Validation
| Script | What It Does | Paper Output |
|--------|--------------|--------------|
| `tenfold_cross_validation.py` | 10‑fold cross‑validation on all six datasets with automatic MW stratification. Full datasets. | **Table 8**, **Table 9** |
| `tenfold_cross_validation_chon.py` | 10‑fold cross‑validation restricted to CHON molecules only. | **Table 9** |

### Dictionary Transferability & Stability
| Script | What It Does | Paper Output |
|--------|--------------|--------------|
| `half_split_transfer.py` | Randomly splits dataset into A and B. Trains on each, evaluates on both (4 pathways). Tests dictionary transferability. | **Table 12**, **Table 13** |
| `synthetic_closure.py` | Trains on A, predicts B, uses predictions as synthetic B', retrains, compares direct vs synthetic pathways. Tests dictionary regenerability. | **Table 14**, **Table 15**, **Table 16** |
| `split_half_stability.py` | Compares bond coefficients from dictionaries trained on independent splits (A vs B). Calculates Δ per bond type. | **Table 17**, **Figure 4** |
| `split_half_stability_figure.py` | Generates Figure 4: two-panel figure showing bond-overlap coefficient differences for Davis dataset. | **Figure 4** |

### Dataset Comparison & Intersection Analysis
| Script | What It Does | Paper Output |
|--------|--------------|--------------|
| `four_way_intersection.py` | Finds overlapping CHON molecules across Taniguchi, He, Davis, Jin. Generates overlap CSVs for further analysis. | **Table 18**, **Table 20**, **Table 21** |
| `four_way_intersection_one_shot.py` | One-shot 4-way intersection validation. Builds 4x4 MAE confusion matrix for Taniguchi, He, Davis, Jin. | **Table 20** |
| `cross_dataset_validation.py` | Trains on Davis, tests on Taniguchi and vice versa. | **Table 19** |
| `within_dataset_validation.py` | Trains and tests on the same dataset (Davis→Davis, Taniguchi→Taniguchi). | **Table 19** |
| `davis_taniguchi_figure.py` | Generates Figure 5: density distributions + correlation scatter plot. | **Figure 5** |

### Polymorphs & Benchmarking
| Script | What It Does | Paper Output |
|--------|--------------|--------------|
| `polymorph_comparison.py` | Compares predicted density against experimental polymorphs (ROY, carbamazepine, aspirin, paracetamol, sulfathiazole, ritonavir). Generates Figure 6 and saves CSV results. | **Figure 6** |
| `benchmark_within_dataset.py` | Trains on each dataset individually and predicts the same dataset (within‑dataset performance). Outputs MAE, RMSE, and R² for all six datasets. These results are used in Table 23 to compare against published models. | **Table 23** |
| `inference_throughput_benchmark.py` | Measures inference throughput vs molecular weight on the combined dataset (229,409 molecules, 9 cores). Splits molecules into 5 MW bins and reports speed per bin. | **Table 22** |

### Supporting / Validation
| Script | What It Does | Paper Output |
|--------|--------------|--------------|
| `ols_vs_nnls_comparison.py` | Compares NNLS vs OLS dictionary optimisation on Taniguchi dataset. Confirms NNLS is physically interpretable without loss of accuracy. | Supplementary (Section 2.3) |
| `co_crystal_demo.py` | (Optional) Demonstrates co‑crystal prediction using mass‑weighted averaging. | Not in paper |

## Quick Start

# Install dependencies
pip install numpy pandas rdkit scikit-learn tqdm matplotlib
