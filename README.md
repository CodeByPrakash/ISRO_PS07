# ISRO BAH 2026 - PS07: AI-Enabled Exoplanet Detection

> Team: **WarpDrive** | Government College of Engineering Kalahandi

---

## Quick Links

| Document | Description |
|----------|-------------|
| [Research Document](ISRO_BAH_2026_Research_Document_PS7_WarpDrive.md) | Full methodology, literature review, architecture |
| [Prototype Notebook](TESS_Exoplanet_ML_Prototype.ipynb) | Working ML model on real xCTL v8.01 data |
| [PPT Structure](AI-Enabled%20Exoplanet%20Detection%20-%20ISRO%20BAH%202026.pptx) | Slide-by-slide content guide |

---

## Problem Statement

**AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves**

- Detect periodic transit signals in TESS data
- Classify: planet candidate / eclipsing binary / stellar blend / starspot
- Estimate parameters: period, duration, depth with uncertainty

---

## Our Approach

| Stage | Tool | Purpose |
|-------|------|---------|
| Data | `lightkurve` + xCTL v8.01 | TESS light curves + 9.5M catalog records |
| Detrend | `wotan` + CBVCorrector | Remove systematics, preserve transits |
| Search | `transitleastsquares` | 93% detection vs 76% BLS |
| Classify | XGBoost + RF + GB ensemble | 46 physics-informed features |
| Fit | `batman` + `emcee` MCMC | Transit parameters + uncertainties |
| Output | SNR + confidence + plots | Publication-ready results |

**Key innovation:** Uses `contratio` (contamination ratio) from TIC v8.1 as ML feature — directly addresses crowded-field blending.

---

## Prototype Results

- **Dataset:** Real 497MB xCTL v8.01 (9.5M records)
- **Processing:** Chunk-based streaming (500k-row blocks)
- **Baseline:** XGBoost — 72% accuracy
- **Classes:** `planetcandidate` / `cooldwarfs_v8` / `hotsubdwarfs_v8`
- **Challenge:** Extreme imbalance — `hotsubdwarfs_v8` = 0.03%

---

## Team

| Role | Name | College |
|------|------|---------|
| Team Lead | Omprakash Behera | GCE Kalahandi |
| Member 1 | Anish Prasad | GCE Kalahandi |

---

## Submission

- **Hackathon:** ISRO BAH 2026
- **Problem:** PS-07
- **Status:** Prototype validated on real TESS data
