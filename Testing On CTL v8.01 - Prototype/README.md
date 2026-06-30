# TESS Exoplanet Classification using xCTL v8.01

> **Hack2Skill × ISRO Hackathon** — Machine Learning for Stellar Object Classification  
> Using the TESS Candidate Target List (CTL v8.01) — a real 497 MB astronomical catalog

---

##  Project Overview

This project applies machine learning to classify stellar objects from NASA's TESS (Transiting Exoplanet Survey Satellite) mission candidate target list. The goal is to automatically distinguish between:

| Class | Label | Description |
|-------|-------|-------------|
|  `PLANET` | `planetcandidate` | Stars likely hosting transiting exoplanets |
|  `COOL_DWARF` | `cooldwarfs_v8` | M-type cool dwarf stars (low-mass, low-temperature) |
|  `HOT_SUBDWARF` | `hotsubdwarfs_v8` | Compact post-red-giant hot subdwarf stars |

The classifier assists ISRO/NASA mission planning by automatically triaging millions of TESS targets, enabling scientists to prioritize the most promising planet-hosting candidates.

---

##  Dataset: xCTL v8.01

| Property | Value |
|----------|-------|
| Source | TESS Candidate Target List v8.01 cross-matched with TIC v8.1 |
| File | `exo_CTL_08.01.csv` |
| Size | ~497 MB |
| Total records | ~9,488,282 |
| Schema | `[ID]:Integer, [priority]:Float, [splists]:String, [objID]:Integer` |

### Class Distribution (Full Catalog)

```
planetcandidate   →  5,545,136   (58.4%)
cooldwarfs_v8     →  3,940,291   (41.5%)
hotsubdwarfs_v8   →      2,855   ( 0.03%)  ← severely imbalanced
```

>  **Class Imbalance**: `hotsubdwarfs_v8` represents only ~0.03% of the dataset — a key challenge addressed in the ML pipeline.

### Training Sample

A stratified 1% sample (~94,883 records) was extracted from the full catalog for model training and evaluation, preserving the class distribution ratio via chunk-based sampling.

---

##  ML Pipeline

### Stage 1 — Data Extraction (`model.ipynb`)
- Streams the 497 MB CSV in 500k-row chunks to avoid memory overflow
- Counts category distributions across all 9M rows
- Extracts a stratified 1% sample: `training_sample.csv`

### Stage 2 — Baseline Model (`model.ipynb`)
- **Random Forest Classifier** (200 trees, single feature: `priority`)
  - Accuracy: 61%, Weighted F1: 0.57
  - Limitation: `hotsubdwarfs_v8` completely missed (F1=0.00) due to extreme imbalance

### Stage 3 — Improved Model (`model.ipynb`)
- **XGBoost Classifier** (300 trees, GPU-accelerated via CUDA)
  - Accuracy: 72%, Weighted F1: 0.72
  - Better separation of `planetcandidate` vs `cooldwarfs_v8`
  - Saved to: `star_classifier.pkl`, `label_encoder.pkl`

### Stage 4 — Full Prototype (`TESS_Exoplanet_ML_Prototype.py`) - underworking
An advanced, physics-informed end-to-end prototype featuring:
- Synthetic TIC v8.1 feature cross-match (stellar physics + photometry)
- Physics-informed feature engineering (color indices, evolutionary indicators)
- Ensemble model: **Random Forest + Gradient Boosting** (soft voting)
- 5-fold cross-validation with F1-Macro scoring
- Full visualization dashboard + saved prediction results

---

##  Feature Engineering

The prototype uses **46 features** across 4 categories:

### Base Catalog Features (from TIC v8.1)
| Group | Features |
|-------|---------|
| Astrometry | `ra`, `dec`, `pmRA`, `pmDEC`, `plx` + uncertainties |
| Photometry | `Tmag`, `Vmag`, `Jmag`, `Hmag`, `Kmag`, `w1mag`, `w2mag`, `GAIAmag` + uncertainties |
| Stellar Physics | `Teff`, `logg`, `rad`, `mass`, `rho`, `lum`, `MH` + uncertainties |
| Contamination | `contratio`, `numcont`, `prox` |
| Quality Flags | `raddflag`, `wdflag`, `starchareFlag` |

### Derived Physics-Informed Features
| Feature | Physical Meaning |
|---------|-----------------|
| `J_K` | Near-infrared color index — cool star indicator |
| `V_K` | Optical-infrared color — hot star indicator |
| `w1_w2` | WISE thermal color — dust/contamination proxy |
| `Teff_logg` | Combined evolutionary state indicator |
| `rad_mass_ratio` | Stellar density proxy |
| `lum_Teff4` | Stefan-Boltzmann law consistency check |
| `contratio_x_prox` | Crowding severity index |
| `log_contratio` | Log-scale contamination ratio |
| `photometric_quality` | Signal-to-noise proxy |
| `pm_quality` | Proper motion astrometric quality |

### Binary Flag Features
`is_giant`, `is_bright`, `high_contamination`, `is_m_dwarf`, `is_hot_star`, `is_subgiant`

---

##  File Structure

```
Testing On CTL v8.01/
│
├── exo_CTL_08.01.csv          # Full TESS CTL v8.01 catalog (497 MB)
├── exo_CTL_08.01_header.csv   # Column schema definition
├── training_sample.csv        # Stratified 1% sample (~95k rows, 5 MB)
├── category_counts.csv        # Full catalog class distribution stats
├── predictions.csv            # Model output predictions
├── unknown.csv                # Unlabeled test records
├── star_classifier.pkl        # Saved XGBoost classifier (~1.8 MB)
├── label_encoder.pkl          # Saved LabelEncoder (class name mapper)
├── model.ipynb                # Jupyter: EDA + baseline RF + XGBoost training
├── TESS_Exoplanet_ML_Prototype.py  # Full advanced ML prototype (713 lines)
├── data_output_extract.py     # Utility: extract unique categories from CSV
├── README.md                  # This file
└── REPORT.md                  # Detailed technical report
```

---

## Installation & Setup

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (optional, for XGBoost GPU acceleration)

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### Run the Baseline Notebook

Open `model.ipynb` in Jupyter and run all cells in sequence:
1. **Cell 1** — Category distribution analysis
2. **Cell 2** — Stratified sample extraction
3. **Cell 3** — Random Forest baseline
4. **Cell 4** — XGBoost improved model
5. **Cell 5** — Model inference on unknown data
---

## Results Summary

### XGBoost (on 1% sample, single feature: `priority`)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| planetcandidate | 0.69 | 0.60 | 0.64 | 7,881 |
| hotsubdwarfs_v8 | 0.00 | 0.00 | 0.00 | 6 |
| cooldwarfs_v8 | 0.74 | 0.81 | 0.77 | 11,090 |
| **Accuracy** | | | **0.72** | 18,977 |

### Prototype Ensemble (multi-feature, synthetic data)

| Metric | Score |
|--------|-------|
| 5-Fold CV F1-Macro | ~0.85 ± 0.02 |
| Classes | 3 (PLANET, COOL_DWARF, HOT_SUBDWARF) |
| Features used | 46 (base + engineered) |

---

##  Known Limitations & Future Work

| Issue | Status | Proposed Fix |
|-------|--------|-------------|
| `hotsubdwarfs_v8` class imbalance (only 2,855 records) | Active | SMOTE oversampling / class weighting |
| Single-feature baseline (priority only) | Addressed in prototype | Full TIC cross-match features |
| GPU memory for 9M records training | Planned | Incremental XGBoost / Dask ML |
| Validation on real planet candidates | Planned | Cross-match with confirmed NASA exoplanet archive |

---

## Context: TESS Mission

TESS (Transiting Exoplanet Survey Satellite) is a NASA mission that surveys ~85% of the sky searching for exoplanets via the transit method. The **CTL (Candidate Target List)** is a prioritized list of ~9.5M stars most amenable to planet detection, generated using TIC (TESS Input Catalog) v8.1 stellar parameters.

This project targets **Problem Statement 7 (PS7)** of the Hack2Skill × ISRO Hackathon, focused on intelligent classification of TESS targets to accelerate planet discovery.

---

## Author

- **Hackathon**: Hack2Skill × ISRO  
- **Track**: Space Technology / Exoplanet Research  
- **Dataset Version**: xCTL v8.01 (TIC v8.1 cross-match)

---

## License

This project is developed for the Hack2Skill × ISRO Hackathon. All astronomical data is sourced from publicly available NASA/MAST archives.

---

*"Every star in the CTL is a potential window to another world."*
