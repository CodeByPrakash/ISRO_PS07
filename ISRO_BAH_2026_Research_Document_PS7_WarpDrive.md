
================================================================================
                    RESEARCH DOCUMENT
    AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves
                    ISRO BAH 2026 | Problem Statement 7
                    Team: WarpDrive
================================================================================

TABLE OF CONTENTS
-----------------
1. Executive Summary
2. Problem Statement Analysis
3. Literature Review & State-of-the-Art (2024-2026)
4. TESS Data Architecture & Catalog Structure
5. Proposed Methodology
6. Technical Architecture
7. Implementation Plan
8. Expected Outcomes & Validation Strategy
9. References

================================================================================
SECTION 1: EXECUTIVE SUMMARY
================================================================================

PROJECT OVERVIEW:
-----------------
Develop an AI-based data analysis pipeline capable of automatically detecting 
exoplanet transit signals from noisy astronomical light curve data from NASA's 
Transiting Exoplanet Survey Satellite (TESS).

KEY CHALLENGE:
--------------
TESS monitors 20,000-30,000 stars per sector, producing massive data volumes.
Transit signals are extremely faint (0.01% to 1% brightness dips) and easily 
confused with:
  - Eclipsing binary stars (V-shaped dips)
  - Stellar blends from crowded fields (asymmetric dips)
  - Starspots and stellar activity (irregular dips)
  - Instrumental systematics (periodic artifacts)

CURRENT GAP (As of 2026):
-------------------------
- TESS has identified 7,821 planet candidates but only ~720 confirmed (as of early 2026)
- ~7,000+ candidates await validation — manual vetting is infeasible at this scale
- Existing pipelines (BLS-based) miss ~24% of small planet signals
- No existing pipeline integrates detection + classification + parameter estimation 
  with uncertainty quantification in a single automated workflow

OUR SOLUTION:
-------------
An end-to-end AI pipeline that:
  1. Downloads and preprocesses TESS light curves using advanced detrending
  2. Searches for periodic signals using Transit Least Squares (TLS) — 93% recovery vs 76% for BLS
  3. Extracts physics-informed shape features (U-shape vs V-shape index)
  4. Classifies signals into 4 categories using ensemble ML (XGBoost + LightGBM)
  5. Fits physical transit models with MCMC for parameter estimation
  6. Provides SNR-based confidence scoring and publication-ready visualizations

UNIQUE VALUE PROPOSITION:
-------------------------
- Uses BOTH light curve shape AND stellar catalog metadata (9.5GB xCTL×TIC v8.1)
- Addresses "crowded field" challenge using pre-computed contamination metrics
- Physics-informed classification: U-shape = planet, V-shape = binary, asymmetric = blend
- Complete uncertainty quantification via MCMC posterior sampling
- Modular, reproducible, open-source pipeline

================================================================================
SECTION 2: PROBLEM STATEMENT ANALYSIS
================================================================================

2.1 REQUIREMENTS BREAKDOWN
--------------------------
The problem statement specifies 6 key algorithmic requirements:

[R1] Identify datasets with periodic dips mimicking astrophysical phenomena
     → Period detection via TLS/BLS periodogram analysis
     → Signal Detection Efficiency (SDE) thresholding

[R2] Classification framework: transits, eclipses, blends, other astrophysical categories
     → 4-way classifier: Planet | Eclipsing Binary | Blend | Stellar Activity
     → Feature engineering: shape parameters + catalog metadata

[R3] Apply classifier on science datasets and correctly categorize signals
     → Batch processing of 20-30k light curves per sector
     → Cross-validation on curated training set

[R4] Provide SNR or significance levels of identified events
     → SNR = Transit Depth / (RMS_noise × √N_transits)
     → SDE from TLS periodogram
     → False Alarm Probability (FAP) estimation

[R5] Estimate transit parameters: depth, period, duration
     → Physical model fitting: Mandel-Agol transit model (batman package)
     → MCMC sampling (emcee) for posterior distributions
     → Best-fit values + 1σ, 2σ confidence intervals

[R6] Visualization of light curves with detected and classified signals
     → Phase-folded plots with model overlays
     → BLS/TLS periodograms with SDE thresholds
     → Corner plots for parameter posteriors
     → Interactive dashboard (Plotly Dash)

2.2 KEY PARAMETERS
------------------
1. Transit Depth (δ): δ = (Rₚ/R★)²
   - Jupiter-sized: ~1% (10,000 ppm)
   - Neptune-sized: ~0.1% (1,000 ppm)
   - Earth-sized: ~0.01% (100 ppm)

2. Transit Duration (T): Time from ingress to egress
   - Typical: 2-4 hours for hot Jupiters
   - Depends on orbital period, stellar radius, impact parameter

3. Orbital Period (P): Interval between consecutive transits
   - Range: 0.5 days (ultra-short period) to 100+ days
   - TESS typically observes each sector for 27 days

4. Signal-to-Noise Ratio (SNR):
   - SNR = δ / σ_noise × √N_transits
   - Detection threshold: SNR > 7-10 for reliable detection

2.3 FALSE POSITIVE SCENARIOS
----------------------------
| Scenario          | Shape    | Key Features                    | Discriminator |
|-------------------|----------|----------------------------------|---------------|
| Planet Transit    | U-shape  | Flat bottom, gradual ingress     | No secondary  |
|                   |          | egress, periodic                 | eclipse       |
| Eclipsing Binary  | V-shape  | Pointed bottom, sharp edges      | Secondary     |
|                   |          | equal depth events               | eclipse present|
| Stellar Blend     | Asymmetric| Irregular, depth varies with     | High contratio|
|                   |          | aperture, contaminated baseline  | from catalog  |
| Starspots/Activity| Irregular| Aperiodic or rotation-modulated  | Stellar var   |
|                   |          | shallow dips                     | flags in TIC  |
| Instrumental      | Periodic | Non-astrophysical pattern        | Quality flags |
|                   |          | spacecraft systematics             | correlation   |

================================================================================
SECTION 3: LITERATURE REVIEW & STATE-OF-THE-ART (2024-2026)
================================================================================

3.1 FOUNDATIONAL WORKS
-----------------------

[1] Charbonneau et al. (2000) — First transiting exoplanet (HD 209458b)
    Established the transit photometry method.

[2] Kovács et al. (2002) — Box Least Squares (BLS) Algorithm
    The standard periodogram method for transit detection.
    Approximates transit as a boxcar function.
    Limitation: Ignores limb darkening and ingress/egress.

[3] Hippke & Heller (2019) — Transit Least Squares (TLS)
    "Transit Least Squares: An optimized transit detection algorithm to search 
    for periodic transits of small planets" (A&A 623, A39)

    KEY FINDINGS:
    - TLS models actual transit shape with limb darkening
    - Analyzes entire unbinned phase-folded light curve
    - Detection efficiency: 93.1% for TLS vs 75.7% for BLS (at 1% FAP)
    - False negative rate: 6.9% for TLS vs 24.3% for BLS
    - Computationally optimized: ~10s for typical K2 light curve on laptop
    - Python implementation: pip install transitleastsquares

    CRITICAL FOR OUR PIPELINE: TLS is essential for detecting small planets
    in noisy TESS data where BLS would miss ~24% of signals.

[4] Hippke & Heller (2019) — GitHub: hippke/tls
    Open-source implementation with tutorials and documentation.
    Dependencies: NumPy, numba, batman-package, tqdm.

3.2 MACHINE LEARNING FOR EXOPLANET DETECTION
--------------------------------------------

[5] Shallue & Vanderburg (2018) — AstroNet
    "Identifying exoplanets with deep learning: A five-planet resonant chain 
    around Kepler-80 and an eighth planet around Kepler-90" (AJ 155, 94)

    - First successful application of CNNs to exoplanet detection
    - Trained on phase-folded Kepler light curves
    - Discovered two new planets including Kepler-90i
    - Training time: several thousand CPU hours
    - Limitation: Performance drops at low SNR due to limited training data

[6] Osborn et al. (2020) — Rapid TESS Classification
    "Rapid classification of TESS planet candidates with convolutional neural 
    networks" (A&A 633, A53)

    - CNN trained for TESS candidate vetting
    - Fast classification but requires large training sets
    - Significant false positive fraction reported

[7] Pearson et al. (2018) — AI Comparison Study
    "Searching for exoplanets using artificial intelligence" (MNRAS 474, 478)

    KEY FINDING: BLS has smallest false negative rate (5%) compared to 
    random forest and CNN classifiers (11-14% false negatives).

    IMPLICATION: Hybrid approach needed — TLS for detection, ML for classification.

[8] Hsu et al. (2018) — Machine Learning Transit Detection
    "A machine learning method to identify Kepler planetary transits" (AJ 155, 251)

    - Systematic comparison of ML methods for transit detection
    - Random forests and CNNs produce significant false positives
    - Recommendation: Use physical models + ML for robust classification

3.3 2024-2026 STATE-OF-THE-ART
--------------------------------

[9] ExoNet (2026) — Multimodal Deep Learning for TESS
    "Multimodal Deep Learning for TESS Exoplanet Candidate Identification via 
    Phase-Folded Light Curves, Stellar Parameters, and Multi-Head Attention 
    Fusion" (arXiv:2604.15560, April 2026)

    KEY INNOVATIONS:
    - Trimodal architecture: 1D CNN + Multi-Head Attention + residual fusion
    - Jointly encodes: (1) global phase-folded view, (2) local transit view, 
      (3) stellar parameter features
    - Validation AUC = 0.8566, Test AUC = 0.8112 on Kepler data
    - Applied to 200 unconfirmed TESS candidates
    - Identified 35 high-confidence signals (≥70%), 27 at >85%
    - 19 candidates in habitable zone (200-400 K equilibrium temperature)
    - Notable: TOI-7949.87 — 99.64% confidence, 0.97 R⊕, Sun-like host
    - Mixed-precision L4 GPU training: <2 hours
    - Code and catalog openly released

    CRITICAL INSIGHT: Multimodal fusion (light curve + stellar params) 
    outperforms single-modality approaches. This validates our strategy 
    of using xCTL×TIC catalog metadata.

[10] Audenaert et al. (2024-2025) — Causal Representation Learning
     "From raw light to astrophysical insight: a causal representation model 
     for TESS light curves" (Astroinformatics 2024, Patagonia)

     - Causal Foundation Models for astronomy
     - Unsupervised learning for stellar variability classification
     - TESS + Kepler + PLATO mission applications
     - Invited talks at IAU General Assembly 2024, Aspen Center for Physics 2024

[11] Frontiers in Astronomy (2026) — Unsupervised ML for TESS
     "Detection of exoplanets from TESS imaging data using unsupervised 
     machine learning techniques" (Frontiers in Space Sciences, May 2026)

     - Modular pipeline: preprocessing → feature extraction → dimensionality 
       reduction → clustering → evaluation
     - Feature categories: basic (time-domain stats) + extended (periodogram)
     - Low-pass filtering for noise suppression
     - Z-score normalization for cross-target comparison
     - Time-aware interpolation for gap filling

[12] Sengupta (2026) — BLS Parameter Estimation Study
     "Detection and Characterization of Exoplanet Transits in Kepler Light 
     Curves" (NHSJS, February 2026)

     - Quantitative analysis of BLS accuracy for parameter estimation
     - Rₚ/R★ ratio estimation with <1% deviation
     - Validates BLS as effective for parameter estimation under good conditions
     - Key parameters: P₀, q (fractional duration), L (in-transit flux), 
       H (out-of-transit flux), t₀ (epoch)

3.4 DETRENDING & NOISE REMOVAL
-------------------------------

[13] Vanderburg & Johnson (2014) — Kepler/K2 Detrending
     "A technique for extracting highly precise photometry for the two-wheeled 
     Kepler mission" (PASP 126, 948)

     - Basis for modern TESS detrending methods
     - Systematic removal for spacecraft motion

[14] Wotan Package (2019+) — Advanced Detrending
     "Trend filtering for astronomical time series" (Hippke et al.)

     - Savitzky-Golay, sliding median, biweight filters
     - Optimized for transit preservation during detrending
     - pip install wotan

[15] Melton et al. (2023) — Autocorrelation Challenge
     ~36% of TESS light curves show significant short-memory autocorrelation 
     after standard detrending. This can mimic transit signals and fool BLS.

     SOLUTION: ARIMA modeling + Transit Comb Filter (TCF) periodogram 
     (Caceres et al. 2019) outperforms BLS for autocorrelated noise.

3.5 TRANSIT MODELING & PARAMETER ESTIMATION
-------------------------------------------

[16] Mandel & Agol (2002) — Analytic Transit Light Curves
     "Analytic light curves for planetary transit searches" (ApJ 580, L171)

     - Foundation of all modern transit fitting
     - Quadratic limb-darkening law
     - Implemented in batman package (Kreidberg 2015)

[17] Kreidberg (2015) — batman: Basic Transit Model
     Python package for fast transit light curve modeling.
     pip install batman-package

     Features:
     - Mandel-Agol analytic model
     - Quadratic, logarithmic, exponential limb-darkening
     - Eclipse and phase curve modeling
     - Fast C implementation with Python wrapper

[18] Maxted (2016) — ellc: Eclipsing Binary Light Curves
     "ellc: A fast, flexible light curve model for detached eclipsing binary 
     stars and transiting exoplanets" (A&A 591, A111)

     - More general than batman: handles eccentric orbits, reflection, Doppler
     - Useful for eclipsing binary classification

[19] Foreman-Mackey et al. (2013) — emcee: MCMC Hammer
     "emcee: The MCMC Hammer" (PASP 125, 306)

     - Ensemble MCMC sampler for parameter estimation
     - Affine-invariant sampling algorithm
     - Standard tool for transit parameter uncertainty quantification
     - pip install emcee

3.6 KEY INSIGHTS FROM LITERATURE
---------------------------------

1. DETECTION: TLS > BLS for small planets (93% vs 76% recovery)
   → Use TLS as primary period search, BLS as backup

2. CLASSIFICATION: Single-modality ML (CNN on light curves only) has 
   high false positive rates (11-14%)
   → Multimodal approach (light curve + stellar params) is superior
   → ExoNet 2026 achieves AUC ~0.85 with trimodal fusion

3. DETRENDING: ~36% of TESS light curves retain autocorrelation after 
   standard detrending, creating false BLS peaks
   → Need ARIMA + TCF or advanced GP-based detrending

4. PARAMETER ESTIMATION: Physical model fitting (batman) + MCMC (emcee) 
   provides robust uncertainties
   → BLS alone gives <1% deviation for Rₚ/R★ but lacks uncertainty quantification

5. CROWDED FIELDS: Contamination is major challenge for TESS
   → xCTL catalog provides pre-computed contratio, numcont metrics
   → Use these as features in classifier, not just post-hoc filtering

================================================================================
SECTION 4: TESS DATA ARCHITECTURE & CATALOG STRUCTURE
================================================================================

4.1 DATA SOURCES
----------------

PRIMARY: TESS Full Frame Images (FFI) and 2-minute Target Pixel Files
- MAST Archive: https://archive.stsci.edu/tess/
- Publicly available, no proprietary period
- Sector-based organization (Sectors 1-56+ as of 2026)
- Two cadences: 2-minute (targeted) and 20-second (FFI)

SECONDARY: TESS Input Catalog (TIC) v8.1 + Exoplanet CTL (xCTL) v8.01
- Standalone xCTL: 497 MB CSV (~70 columns)
- Cross-matched xCTL×TIC v8.1: 9.5 GB CSV (~200+ columns)
- 3.5 million targets with stellar parameters
- Header file: exo_CTL_08.01xTIC_v8.1_header.csv

4.2 CRITICAL COLUMNS FROM xCTL×TIC v8.1
----------------------------------------

IDENTIFIERS & ASTROMETRY:
- ID: TIC identifier (primary key)
- ra, dec: J2000 coordinates (degrees)
- e_RA, e_Dec: Coordinate uncertainties
- pmRA, pmDEC: Proper motions (mas/yr)
- e_pmRA, e_pmDEC: Proper motion uncertainties
- plx, e_plx: Parallax and uncertainty
- gallong, gallat: Galactic coordinates
- eclong, eclat: Ecliptic coordinates

PHOTOMETRY (Multi-Wavelength):
- Tmag, e_Tmag: TESS magnitude and uncertainty (PRIMARY)
- Bmag, Vmag: Johnson-Cousins optical
- umag, gmag, rmag, imag, zmag: SDSS photometry
- Jmag, Hmag, Kmag: 2MASS near-infrared
- w1mag, w2mag, w3mag, w4mag: WISE thermal infrared
- GAIAmag, e_GAIAmag: Gaia DR3 G-band magnitude
- gaiabp, gaiarp: Gaia BP and RP magnitudes

STELLAR PHYSICS (For Transit Models):
- Teff, e_Teff: Effective temperature (K) → Limb-darkening coefficients
- logg, e_logg: Surface gravity (cm/s²) → Stellar density, evolutionary state
- rad, e_rad: Stellar radius (R☉) → Planet radius from transit depth
- mass, e_mass: Stellar mass (M☉) → Kepler's 3rd law validation
- rho, e_rho: Stellar density (ρ☉) → Transit duration constraint
- lum, e_lum: Luminosity (L☉) → Habitable zone estimation
- MH, e_MH: Metallicity [M/H] → Planet formation probability
- lumclass: Luminosity class (dwarf/giant) → Giant = false positive risk

CONTAMINATION & CROWDING (CRITICAL FOR PS7):
- contratio: Contamination ratio (fraction of flux from nearby sources)
- numcont: Number of contaminating sources in aperture
- prox: Distance to nearest neighbor (arcseconds)
- numcont: Number of nearby contaminants
- contratio: Critical threshold >0.1 indicates significant blending

QUALITY FLAGS:
- disposition: Training labels (PC=Planet Candidate, FP=False Positive, 
  EB=Eclipsing Binary, etc.)
- priority: TESS observation priority score
- raddflag: Radius discrepancy flag (0=OK, 1=unreliable)
- starchareFlag: Stellar characterization quality
- wdflag: White dwarf flag (special stellar type)
- duplicate_id: Duplicate target identifier
- TESSflag, SPFlag, PMflag, PARflag: Various quality flags

ASTROMETRIC QUALITY (Gaia DR3):
- gaiaqflag: Gaia astrometric quality flag
- RUWE (implied): Renormalized Unit Weight Error — high values indicate 
  astrometric noise from unresolved companions

4.3 HOW WE USE THE CATALOG IN OUR PIPELINE
-------------------------------------------

STEP 1: TARGET SELECTION
- Filter: disposition IN ('PC', 'FP', 'EB') for training labels
- Quality cuts: raddflag=0, starchareFlag='OK', e_Tmag<0.1, wdflag=0
- Remove: Giants (logg < 3.5) for planet search (too large for small transits)
- Priority: High priority targets for focused analysis

STEP 2: STELLAR PRIORS FOR TRANSIT MODELING
- Teff → Look up limb-darkening coefficients (u1, u2) from Claret tables
- rad, mass → Compute expected stellar density ρ★ = M★/R★³
- Compare with transit-derived density: consistency check for planet validation
- logg → Surface gravity constraint on transit duration

STEP 3: NOISE PREDICTION
- Tmag → Estimate photometric noise: σ(T) = f(Tmag) ppm/hour
- Used for SNR estimation and detection threshold setting
- Brighter stars (Tmag < 12) → Lower noise → Higher SNR

STEP 4: CONTAMINATION ASSESSMENT
- contratio > 0.1: Flag as "High Blend Risk"
- numcont > 2: Exclude from planet search or label as "Blend"
- prox < 30 arcsec: Aperture crowding warning
- These become FEATURES in our ML classifier, not just filters

STEP 5: TRAINING DATA CONSTRUCTION
- disposition = "PC" → Positive label (Confirmed Planet)
- disposition = "FP" → Negative label (False Positive)
- disposition = "EB" → Negative label (Eclipsing Binary)
- Cross-match with light curves for multimodal training

4.4 DATA VOLUME ESTIMATES
--------------------------
- Per sector: 20,000-30,000 light curves (2-minute cadence)
- Full mission (56 sectors): ~1.5 million light curves
- xCTL catalog: 3.5 million targets × 200+ columns = 9.5 GB
- Light curve data: ~1-10 MB per star per sector
- Total data volume: ~10-100 TB for full mission

================================================================================
SECTION 5: PROPOSED METHODOLOGY
================================================================================

5.1 PIPELINE OVERVIEW (8 STAGES)
--------------------------------

Stage 1: DATA INGESTION
- Download TESS 2-minute cadence PDCSAP flux from MAST
- Cross-match with xCTL×TIC v8.1 for stellar parameters
- Quality flag masking (remove bad data points)
- Output: Clean time series + stellar metadata

Stage 2: PREPROCESSING & DETRENDING
- Savitzky-Golay filtering (wotan package) for long-term trends
- CBVCorrector (lightkurve) for systematic noise removal
- Sigma-clipping (5σ) for outlier rejection
- Gap interpolation using time-aware methods
- Output: Flattened light curve with clean baseline

Stage 3: PERIODIC SIGNAL SEARCH
- Primary: Transit Least Squares (TLS) for small planet sensitivity
- Backup: Box Least Squares (BLS) for comparison
- SDE threshold: SDE > 10 for significant detection
- Period refinement: Fold at best period, verify periodicity
- Output: Candidate periods + SDE values + phase-folded curves

Stage 4: EVENT EXTRACTION
- Phase-fold light curve at detected period
- Extract transit windows (ingress, egress, flat bottom)
- Bin and stack multiple transits for SNR enhancement
- Output: Phase-folded transit profiles + individual transit snippets

Stage 5: FEATURE ENGINEERING

A. SHAPE FEATURES (Physics-Informed):
   - U-shape index: Flatness of transit bottom (planet vs binary)
   - V-shape index: Pointedness of transit bottom (binary indicator)
   - Ingress/egress slope ratio: Symmetry of transit edges
   - Duration ratio: T_transit / P_orbit (physical constraint)
   - Depth asymmetry: Primary vs secondary eclipse depth

B. STATISTICAL FEATURES:
   - Skewness, kurtosis of phase-folded curve
   - Autocorrelation at lag = period
   - Lomb-Scargle power at harmonics

C. CATALOG FEATURES (From xCTL×TIC):
   - contratio, numcont, prox (contamination)
   - Teff, logg, rad, mass (stellar priors)
   - Tmag, e_Tmag (noise prediction)
   - lumclass (giant vs dwarf)

D. FREQUENCY DOMAIN FEATURES:
   - BLS/TLS power spectrum shape
   - Harmonic content (secondary eclipse detection)
   - Periodogram peak width and significance

Stage 6: AI CLASSIFICATION
- Ensemble model: XGBoost (primary) + LightGBM (secondary) + Random Forest (validation)
- Training data: Curated set with disposition labels
- 4-class output: Planet | Eclipsing Binary | Blend | Stellar Activity
- Soft voting: Probability scores for each class
- Threshold optimization: Maximize F1-score per class
- Cross-validation: 5-fold stratified to handle class imbalance

Stage 7: PARAMETER ESTIMATION
- Model: Mandel-Agol transit (batman package)
- Fitting: LMFIT for initial least-squares, then emcee MCMC
- Parameters: P, T₀, δ (depth), a/R★, i (inclination), b (impact parameter)
- Limb darkening: Quadratic law with coefficients from Teff lookup
- Uncertainty: Posterior distributions from MCMC chains
- Convergence: Gelman-Rubin statistic < 1.01

Stage 8: CONFIDENCE SCORING & OUTPUT
- SNR = δ / σ_noise × √N_transits
- SDE from TLS periodogram
- FAP (False Alarm Probability) from bootstrap analysis
- Model confidence: Class probability from ensemble
- Final score: Combined metric weighing SNR + model confidence + FAP
- Output: Classified signal + parameters + uncertainties + plots

5.2 DETAILED ALGORITHMS
-----------------------

5.2.1 DETRENDING ALGORITHM
Algorithm: Wotan Savitzky-Golay + CBV Correction

Input: Raw PDCSAP flux time series f(t), quality flags q(t)
Output: Detrended flux f_det(t), trend model T(t)

1. Mask bad quality points: q(t) = 0 → exclude
2. Initial outlier clip: |f(t) - median(f)| > 5σ → mask
3. CBV correction: 
   - Download cotrending basis vectors for sector/camera/CCD
   - Fit: f(t) = Σᵢ cᵢ × CBVᵢ(t) + residual
   - Corrected: f_cbv(t) = f(t) - Σᵢ cᵢ × CBVᵢ(t)
4. Savitzky-Golay filtering:
   - Window size: 2× transit duration (typically 6-12 hours)
   - Polynomial order: 3
   - Trend: T(t) = SG_filter(f_cbv(t))
   - Detrended: f_det(t) = f_cbv(t) / T(t) - 1 (in ppm)
5. Final outlier clip: |f_det(t)| > 5σ_det → mask
6. Interpolate gaps: Linear interpolation for gaps < 2 hours

5.2.2 PERIOD SEARCH ALGORITHM
Algorithm: Transit Least Squares (TLS)

Input: Detrended flux f_det(t), time t, stellar density ρ★ (optional)
Output: Best period P, SDE, phase-folded curve

1. Grid search over periods:
   - Range: 0.5 days to (T_obs / 2) where T_obs = 27 days (sector length)
   - Sampling: Adaptive (finer near expected periods from stellar density)
2. For each trial period P:
   - Phase-fold: φ = (t - t₀) / P mod 1
   - Sort by phase
   - Fit transit model with limb darkening (u1, u2 from Teff)
   - Duration grid: 0.5 to 12 hours (physically plausible range)
   - Compute χ² of fit
3. Compute SDE (Signal Detection Efficiency):
   - SDE = (χ²_best - mean(χ²)) / std(χ²)
   - Threshold: SDE > 10 for significant detection
4. Refine best period:
   - Zoom-in around peak period ±1%
   - Fine grid search for precise P
5. Output: P, SDE, phase-folded curve at best period

5.2.3 FEATURE EXTRACTION ALGORITHM

A. U-Shape vs V-Shape Index:
   - Fit parabola to transit bottom: y = a(x-x₀)² + b
   - U-shape index: flatness = 1 - |a| / (|a|_max)
   - V-shape index: pointedness = |a| / |a|_max
   - Threshold: U-index > 0.7 → planet; V-index > 0.7 → binary

B. Ingress/Egress Slope Ratio:
   - Linear fit to ingress (first 10% of transit duration)
   - Linear fit to egress (last 10% of transit duration)
   - Ratio = slope_ingress / slope_egress
   - ~1.0 → symmetric (planet); >>1 or <<1 → asymmetric (blend)

C. Secondary Eclipse Detection:
   - Search for dip at phase 0.5 (opposite to transit)
   - Depth ratio: δ_secondary / δ_primary
   - >0.1 → likely eclipsing binary
   - <0.01 → consistent with planet (no secondary)

5.2.4 CLASSIFICATION ALGORITHM
Algorithm: Ensemble Voting Classifier

Input: Feature vector x = [shape_features, statistical_features, catalog_features]
Output: Class probabilities y = [P_planet, P_binary, P_blend, P_activity]

1. XGBoost Classifier:
   - Hyperparameters: max_depth=6, learning_rate=0.1, n_estimators=500
   - Objective: multi:softprob
   - Class weights: Inverse frequency for imbalance

2. LightGBM Classifier:
   - Hyperparameters: num_leaves=31, learning_rate=0.05
   - Feature fraction: 0.8 (prevent overfitting)

3. Random Forest Classifier:
   - Hyperparameters: n_estimators=200, max_depth=10
   - Used for feature importance validation

4. Soft Voting Ensemble:
   - y_ensemble = (y_xgb + y_lgb + y_rf) / 3
   - Final class: argmax(y_ensemble)
   - Confidence: max(y_ensemble)

5.2.5 PARAMETER ESTIMATION ALGORITHM
Algorithm: MCMC Transit Fitting

Input: Phase-folded light curve, initial parameters from TLS
Output: Posterior distributions for P, T₀, δ, a/R★, i, b, u1, u2

1. Initialize parameters from TLS:
   - P₀ = TLS period
   - T₀ = TLS epoch
   - δ₀ = TLS depth
   - a/R★₀ = from Kepler's 3rd law (using stellar mass from catalog)
   - i₀ = 90° (edge-on assumption)
   - b₀ = 0 (central transit assumption)
   - u1, u2 = from Claret tables (using Teff, logg)

2. LMFIT initial fit:
   - Least-squares minimization
   - Fix u1, u2 (from catalog)
   - Free: P, T₀, δ, a/R★, i, b
   - Output: Best-fit parameters + covariance matrix

3. emcee MCMC sampling:
   - 100 walkers, 5000 steps, 20% burn-in
   - Priors:
     * P: Gaussian around LMFIT value, σ = 0.1%
     * T₀: Uniform within one period
     * δ: Uniform [0, 0.1] (physical constraint)
     * a/R★: Gaussian around stellar density constraint
     * i: Uniform [80°, 90°] (transiting geometry)
     * b: Uniform [0, 1] (impact parameter)
   - Likelihood: χ² = Σᵢ (f_obs - f_model)² / σᵢ²

4. Convergence check:
   - Gelman-Rubin R̂ < 1.01 for all parameters
   - Effective sample size > 1000 per parameter

5. Output statistics:
   - Best-fit: median of posterior
   - Uncertainty: 16th and 84th percentiles (±1σ)
   - 95% CI: 2.5th and 97.5th percentiles

5.3 HANDLING THE "CROWDED FIELD" CHALLENGE
-------------------------------------------

The problem statement explicitly mentions:
"significant contaminations arising from effects such as stellar blending 
by foreground or background sources in the aperture"

OUR APPROACH:

1. PRE-FILTERING (Before Processing):
   - contratio > 0.1: Flag as "High Blend Risk"
   - numcont > 2: Exclude from planet search or require manual review
   - prox < 30 arcsec: Warning — aperture contains multiple sources

2. FEATURE ENGINEERING (During Classification):
   - Include contratio, numcont, prox as classifier features
   - High contratio increases probability of "Blend" class
   - Use as continuous features, not binary thresholds

3. POST-PROCESSING (After Detection):
   - For planet candidates with contratio > 0.05:
     * Require higher SNR threshold (>15 instead of >7)
     * Require higher confidence (>90% instead of >70%)
     * Flag for follow-up observation (high-resolution imaging)

4. VALIDATION:
   - Compare transit-derived stellar density (ρ_transit) with 
     catalog density (ρ_catalog = M★/R★³)
   - If |ρ_transit - ρ_catalog| / ρ_catalog > 30% → Blend warning
   - Blended light curves have distorted transit shapes → density mismatch

================================================================================
SECTION 6: TECHNICAL ARCHITECTURE
================================================================================

6.1 SYSTEM ARCHITECTURE (5 LAYERS)
-----------------------------------

LAYER 1: DATA LAYER
┌─────────────────────────────────────────────────────────────┐
│  TESS MAST Archive API          │  xCTL×TIC v8.1 Catalog    │
│  • Sector 1-56 light curves     │  • 9.5 GB CSV             │
│  • 2-min cadence PDCSAP flux    │  • 3.5M targets           │
│  • Target Pixel Files (TPF)     │  • 200+ columns           │
│  • Quality flags                │  • Stellar params + contam │
│  • Local cache (FITS + CSV)     │  • Training labels        │
└─────────────────────────────────────────────────────────────┘

LAYER 2: PROCESSING LAYER
┌─────────────────────────────────────────────────────────────┐
│  Detrending Module    │  Period Search Module   │  Folding   │
│  • Wotan SG filter    │  • TLS (primary)       │  Module    │
│  • CBV correction     │  • BLS (backup)        │  • Phase   │
│  • Outlier rejection  │  • SDE thresholding    │    fold    │
│  • Gap interpolation  │  • Period refinement   │  • Bin &   │
│                       │                        │    stack   │
└─────────────────────────────────────────────────────────────┘

LAYER 3: AI LAYER
┌─────────────────────────────────────────────────────────────┐
│  Training Pipeline      │  Ensemble Classifier    │  Validation │
│  • Curated dataset      │  • XGBoost (primary)   │  • 5-fold CV│
│  • Feature scaling      │  • LightGBM (secondary)│  • Grid search│
│  • Label encoding       │  • Random Forest       │  • F1 opt   │
│  • SMOTE balancing      │  • Soft voting         │  • ROC-AUC  │
└─────────────────────────────────────────────────────────────┘

LAYER 4: ESTIMATION LAYER
┌─────────────────────────────────────────────────────────────┐
│  Transit Model (batman)  │  MCMC Sampling (emcee) │  Params  │
│  • Mandel-Agol analytic   │  • 100 walkers          │  • P, T₀ │
│  • Quadratic LD (u1,u2)  │  • 5000 steps           │  • δ     │
│  • Eclipse modeling       │  • Burn-in 20%          │  • a/R★  │
│  • Reflection/phase       │  • Gelman-Rubin < 1.01  │  • i, b  │
└─────────────────────────────────────────────────────────────┘

LAYER 5: OUTPUT LAYER
┌─────────────────────────────────────────────────────────────┐
│  SNR & Confidence    │  Visualization        │  Reporting    │
│  • SDE > 10          │  • Phase-folded plots │  • 3-page     │
│  • SNR > 7           │  • Model overlays     │    technical  │
│  • FAP < 1%          │  • Corner plots       │    report     │
│  • Model confidence  │  • BLS periodograms   │  • LaTeX/PDF  │
│                      │  • Interactive Dash   │               │
└─────────────────────────────────────────────────────────────┘

6.2 TECHNOLOGY STACK
--------------------

| Category          | Tool/Library        | Version    | Purpose                    |
|-------------------|---------------------|------------|----------------------------|
| Data Access       | lightkurve          | 2.4+       | TESS data I/O              |
|                   | astroquery          | 0.4+       | MAST archive queries       |
|                   | pandas              | 2.0+       | Catalog data manipulation  |
| Detrending        | wotan               | 1.1+       | Trend filtering            |
|                   | CBVCorrector        | built-in   | Systematic removal         |
| Period Search     | transitleastsquares | 1.0+       | TLS algorithm              |
|                   | astropy.timeseries  | 5.0+       | BLS periodogram            |
| Feature Extraction| numpy               | 1.24+      | Numerical arrays           |
|                   | scipy.stats         | 1.10+      | Statistical features       |
|                   | tsfresh             | 0.20+      | Time series features       |
| ML Framework      | xgboost             | 2.0+       | Gradient boosting          |
|                   | lightgbm            | 4.0+       | Gradient boosting          |
|                   | scikit-learn        | 1.3+       | ML utilities, RF, CV       |
|                   | imbalanced-learn    | 0.11+      | SMOTE balancing            |
| Deep Learning     | PyTorch             | 2.0+       | Optional CNN (ExoNet-style)|
| Transit Modeling  | batman-package      | 2.4+       | Mandel-Agol model          |
|                   | lmfit               | 1.2+       | Least-squares fitting      |
|                   | ellc                | 1.8+       | Binary star modeling       |
| MCMC Sampling     | emcee               | 3.1+       | Ensemble MCMC              |
|                   | corner              | 2.2+       | Posterior visualization    |
| Visualization     | matplotlib          | 3.7+       | Static plots               |
|                   | seaborn             | 0.12+       | Statistical plots            |
|                   | plotly              | 5.15+      | Interactive dashboard      |
|                   | plotly-dash         | 2.14+      | Web application            |
| Reporting         | Jupyter             | 7.0+       | Notebooks                  |
|                   | LaTeX               | TeX Live   | Technical report           |
| Infrastructure    | Python              | 3.10+      | Runtime                    |
|                   | CUDA                | 12.0+      | GPU acceleration (optional)|
|                   | Docker              | 24.0+      | Containerization           |

6.3 PIPELINE WORKFLOW DIAGRAM
-----------------------------

[![Complete Pipeline Data Flow](https://lh3.googleusercontent.com/rd-d/ALs6j_GrL_875w4I8uHXpk6FzD5eLWHPbEODHsTLa9NPduERw6pJZVJM0H8EQJJAZ2o6KEkfazZIkrQFzA0BuOf4QG8aH7S8t06AcJD3gHoHqbjeQOy4DJLgwDlfQMyHB13hNE1yegDWDl5kwoGqdcNjge8iRbi1sM-M9jUcQ7zuxx7o5uZj42kBm2Qa9OrMeWhhAsvTlJbO41Nk9zI30G8THGllm6sjPPKuRmvYAKk-KngVrzOUqUhrNJID27NGYs_Lj4GXnf_-EKxm-esFot6L4qbFkFk5FMTpbbTf6LpzY58NHFY5ccjewmAoO8sGMubHQeiNvj6p89T8KHuYcXiBbvqoogkg4UD2mEAg1JBGOWkgDzH7U5P5z5rNgZ9Vto8Ss5JEaLijr60BCwh27uQq2Qzt0wsQZQwKGHXcSlcHLNAX0VBiCy5qUJHeHuJ6b5r4i-4hpdfmZri-CA4QW8_4YhOuFcf5NDBXqTUzNp2NQk1hUjzJnYTN7jtGAgmAvawIztCGVwPfsP-v6gA9CtCWjL9BueiYuOE3hw6Yo6fd6m-B_DmOwTC7DkPvg6_Ju6XhpvcJvSZ-d_iaEK7O9vfUoD_K2ocJCj_7tnOWLR5ezlIBnZzTU51jCnk7VqZZ83Jf6mVb1nUgAsyItEbGbgOOYAxMiV6HD-55LY3zRvT9ywKOt8SaxHMAzjDkCYXPk0HBVape-_PdwN6I0pu3MEd-vlgTpvg7XK7_MSKqfK0pkMErsWgem2cAK9nDoOhgXEUND0mux1vCZ_RDDeyrcO7cVBrpMrpiwvfcIB2QjKa86eMY_J-gynF4CMQrAyUkQfQwJbHtARUd7WInAcSsP6GGJ16YOv88VMRmQ39riSQWedPOuQabJ6biBwcfxDevd7nxhxH0jZm61_UHFVrDaOc-Coce0ObJ_lNaJuPPfZq9eFz1Qe6OAuoJ18ebwnJLSnMLnfpfHk4I58Uiu3yZKwGU_Lo6YJpTxj0of6_dKcuLJaZnV78DJ3uV07kjUx1v0wZhyTbp46YxAECAf_eE6kvB836xtkTk23xj1oCmNYh4u8JLaEFv1dDeue4ex_m-5tG1e8QGiE78aehxtL4WBz1YAYyXYqqtTno=w2000-h1456?auditContext=forDisplay)](https://lh3.googleusercontent.com/rd-d/ALs6j_GrL_875w4I8uHXpk6FzD5eLWHPbEODHsTLa9NPduERw6pJZVJM0H8EQJJAZ2o6KEkfazZIkrQFzA0BuOf4QG8aH7S8t06AcJD3gHoHqbjeQOy4DJLgwDlfQMyHB13hNE1yegDWDl5kwoGqdcNjge8iRbi1sM-M9jUcQ7zuxx7o5uZj42kBm2Qa9OrMeWhhAsvTlJbO41Nk9zI30G8THGllm6sjPPKuRmvYAKk-KngVrzOUqUhrNJID27NGYs_Lj4GXnf_-EKxm-esFot6L4qbFkFk5FMTpbbTf6LpzY58NHFY5ccjewmAoO8sGMubHQeiNvj6p89T8KHuYcXiBbvqoogkg4UD2mEAg1JBGOWkgDzH7U5P5z5rNgZ9Vto8Ss5JEaLijr60BCwh27uQq2Qzt0wsQZQwKGHXcSlcHLNAX0VBiCy5qUJHeHuJ6b5r4i-4hpdfmZri-CA4QW8_4YhOuFcf5NDBXqTUzNp2NQk1hUjzJnYTN7jtGAgmAvawIztCGVwPfsP-v6gA9CtCWjL9BueiYuOE3hw6Yo6fd6m-B_DmOwTC7DkPvg6_Ju6XhpvcJvSZ-d_iaEK7O9vfUoD_K2ocJCj_7tnOWLR5ezlIBnZzTU51jCnk7VqZZ83Jf6mVb1nUgAsyItEbGbgOOYAxMiV6HD-55LY3zRvT9ywKOt8SaxHMAzjDkCYXPk0HBVape-_PdwN6I0pu3MEd-vlgTpvg7XK7_MSKqfK0pkMErsWgem2cAK9nDoOhgXEUND0mux1vCZ_RDDeyrcO7cVBrpMrpiwvfcIB2QjKa86eMY_J-gynF4CMQrAyUkQfQwJbHtARUd7WInAcSsP6GGJ16YOv88VMRmQ39riSQWedPOuQabJ6biBwcfxDevd7nxhxH0jZm61_UHFVrDaOc-Coce0ObJ_lNaJuPPfZq9eFz1Qe6OAuoJ18ebwnJLSnMLnfpfHk4I58Uiu3yZKwGU_Lo6YJpTxj0of6_dKcuLJaZnV78DJ3uV07kjUx1v0wZhyTbp46YxAECAf_eE6kvB836xtkTk23xj1oCmNYh4u8JLaEFv1dDeue4ex_m-5tG1e8QGiE78aehxtL4WBz1YAYyXYqqtTno=w2000-h1456?auditContext=forDisplay)


================================================================================
SECTION 7: IMPLEMENTATION PLAN
================================================================================

7.1 DEVELOPMENT TIMELINE (4-6 Weeks)
------------------------------------

Week 1: Data Infrastructure & Preprocessing
- Day 1-2: Set up MAST API access, download sample sector data
- Day 3-4: Implement xCTL catalog reader (chunked CSV processing)
- Day 5-7: Build detrending module (wotan + CBV correction)

Week 2: Detection & Feature Engineering
- Day 8-10: Implement TLS period search with SDE calculation
- Day 11-12: Implement BLS backup and comparison
- Day 13-14: Build feature extraction pipeline (shape + stats + catalog)

Week 3: AI Classification
- Day 15-17: Curate training dataset from disposition labels
- Day 18-19: Train XGBoost classifier with hyperparameter tuning (Optuna)
- Day 20-21: Build ensemble (XGB + LGBM + RF) with soft voting

Week 4: Parameter Estimation & Validation
- Day 22-24: Implement batman transit model fitting with LMFIT
- Day 25-26: Add emcee MCMC sampling with convergence checks
- Day 27-28: Validate on known exoplanets (compare with NASA archive)

Week 5: Visualization & Dashboard
- Day 29-31: Build static plotting functions (matplotlib)
- Day 32-33: Create interactive dashboard (Plotly Dash)
- Day 34-35: Generate corner plots and periodogram visualizations

Week 6: Testing, Documentation & Report
- Day 36-38: End-to-end pipeline testing on full sector (20-30k LCs)
- Day 39-40: Performance benchmarking (speed, accuracy, memory)
- Day 41-42: Write 3-page technical report (LaTeX)
- Day 43-45: Final validation, bug fixes, documentation

7.2 MILESTONES & DELIVERABLES
-----------------------------

Milestone 1 (Week 1): Data Pipeline Operational
- Deliverable: Preprocessing module with detrending, tested on 100 LCs
- Success metric: Detrended RMS < 2× theoretical noise for >90% of targets

Milestone 2 (Week 2): Detection Working
- Deliverable: TLS search with SDE calculation, tested on known planets
- Success metric: Recover >90% of known planets in test set

Milestone 3 (Week 3): Classifier Trained
- Deliverable: Ensemble model with >85% accuracy on validation set
- Success metric: F1-score >0.85 for all 4 classes

Milestone 4 (Week 4): Fitting Working
- Deliverable: MCMC parameter estimation with uncertainties
- Success metric: Parameter errors <10% for SNR>15 signals

Milestone 5 (Week 5): Dashboard Live
- Deliverable: Interactive web dashboard with all visualizations
- Success metric: Load time <5s for 1000 light curves

Milestone 6 (Week 6): Final Delivery
- Deliverable: Complete pipeline + 3-page report + presentation
- Success metric: Process full sector in <8 hours

7.3 RISK MITIGATION
-------------------

| Risk                          | Mitigation Strategy                          |
|-------------------------------|----------------------------------------------|
| Large data volume (9.5GB CSV) | Chunked reading with pandas; filter by sector  |
|                               | before loading; use Dask for parallelization |
| Long TLS computation time     | Use numba JIT compilation; restrict duration |
|                               | grid; parallelize across CPU cores           |
| Class imbalance in training   | SMOTE oversampling; class weights in XGBoost |
|                               | Stratified sampling; ensemble voting         |
| MCMC non-convergence          | Multiple restart points; Gelman-Rubin check; |
|                               | increase walkers if R̂ > 1.01               |
| False positives from blends   | Use contratio as feature; density validation;|
|                               | higher thresholds for high-blend targets     |
| Memory limitations            | Process light curves sequentially; save      |
|                               | intermediate results to disk; use generators |

================================================================================
SECTION 8: EXPECTED OUTCOMES & VALIDATION STRATEGY
================================================================================

8.1 PERFORMANCE TARGETS
-------------------------

DETECTION PERFORMANCE:
- Recall (True Positive Rate): >90%
  → Don't miss real planets (minimize false negatives)
- Precision: >85%
  → Don't claim blends as planets (minimize false positives)
- F1-Score: >87%
  → Balanced performance
- False Alarm Rate: <1% at SDE > 10

CLASSIFICATION PERFORMANCE:
- Overall Accuracy: >95%
- Per-class F1-Score: >87% for all 4 classes
- AUC-ROC: >0.92 for planet vs. non-planet binary classification

PARAMETER ESTIMATION ACCURACY:
- Orbital Period (P): ±0.1%
- Transit Duration (T): ±5%
- Transit Depth (δ): ±10% (for SNR > 15)
- Epoch (T₀): ±0.01 days
- Impact Parameter (b): ±0.1

PROCESSING SPEED:
- Single light curve (end-to-end): <5 minutes
- Batch processing (1000 LCs): <8 hours
- Full sector (20-30k LCs): <1 week on single workstation

8.2 VALIDATION STRATEGY
-----------------------

1. INJECTION-RETRIEVAL TESTS:
   - Inject synthetic transits into real TESS noise
   - Vary: depth (100-10000 ppm), period (1-30 days), duration (1-4 hours)
   - Measure recovery rate vs. SNR
   - Compare TLS vs BLS detection efficiency

2. KNOWN PLANET VALIDATION:
   - Test on confirmed exoplanets from NASA Exoplanet Archive
   - Compare derived parameters with literature values
   - Verify classification accuracy on labeled dataset

3. CROSS-VALIDATION:
   - 5-fold stratified cross-validation on curated training set
   - Leave-one-sector-out validation (test on unseen TESS sector)
   - Temporal validation (train on early sectors, test on late sectors)

4. BLIND TEST:
   - Process unlabeled TESS sector
   - Compare predictions with follow-up observations (if available)
   - Flag high-confidence candidates for community validation

5. FALSE POSITIVE ANALYSIS:
   - Inject known false positive types (EB, blends, spots)
   - Measure classifier rejection rate
   - Analyze failure modes and feature importance

8.3 SUCCESS METRICS DASHBOARD
-----------------------------

[![Success Metrics and Targets](https://lh3.googleusercontent.com/rd-d/ALs6j_FEjYD-Xywb4HJyAoT_lHXloBT9Zo5vwok1ZABLsj1ac8k5eRBh1bJPNiYtp2yUe--Xota2pEJJVkMCR2T7QGDOQ0nPLw9BLeNZqls-_3yHjPPAEqJpTQZpu7YEYyBvmFjF_62dTjGPkCV4XvbVmTxoIQE0G7KJQHzdx3czzUnLPrp7YICXVApiOibKi_AcPoHUt3qgmiq8H87fTdWYAhJr9xKZCnJzHWo3vy509Mp58n4Qf2rlRml6GXpoiBzT7ojYB6kXULb0MyKDBcMuXawzZpslf_4bAJUj_Tw1cVWHq53kL_OBO6DUcd-e9LG_gKljLlFm3vtGqWwpvyp2ePf7qkwOQTw2kQgEjDgV_LZBEYYCaAT4mI1VMRwZBMFpDN9wKseZQsW9o97TLHvOSNPeAgZEIDurl1EUuj7WtIe6IL2P8rjjPIZ1ItMLUOPCWh4S1CpL6Hg5gj3r7h0GtoRNMdAQcjKSmD1RRp_zwUnJ7VaR2Tl2RHG69kDR4uBGQ_2zs9VDaCfXIZjQ6HDh1IempaZeJaVddyamDjGdGiZCDkjcAqpiVceqrvS5kb-n9fEB4wfNbGqbW_Svj3yIYipIfjL8bni-G3yNt1K2645KL1uZr01rTgQsypFEIh9JtWaU5AWUA_Ssq1AZWszIDWI35Mgic2UMnlg6NtnCOf9Eb7beu0FX4cBIT8hDhRbMhvJJB2YZICBgqiyMXYDx1AZJ4jvKF69juSGAaD-hSAgDs1h6gzM5UHPtrjgLPTBP5oBB5AKc6vBejCBZ4xydNuzijDaNRtlMhXFOUgwvDoF17XCeh0dPr5Dnx2AwnJwgqA6U1eNAWdKjYxcph89JnakO7xD0B8xcHjkiUM6sV_idZBMJp4elydIseMHAxzEghbClOOd8qQmsnIICNXIr48dwy9Rc6HULqoAom1c-O8wTZQZuZbjV5YHVCHlWFFAK4_Gy5lcw66Z_P0R6JWAvgQD8y6lFBS_JwIkYkoDy8zmpNGbMJDntxJVGyf7Te6IdGKnLpdBK_tqnZc7sUTgtgxrGCNDK4XLaEo-iNeahdqq1duI5A-x8kzj-xw6G4vVyJ0YbX25vMn_Dn75vni0ca6CFHLCJ3HI=w1920-h945?auditContext=prefetch)](https://your-target-website.com)


8.4 EVALUATION CRITERIA ALIGNMENT
---------------------------------

ISRO EVALUATION CRITERIA → OUR APPROACH:

1. ACCURACY OF DETECTION & CLASSIFICATION
   → Ensemble ML (XGBoost + LightGBM + RF) with calibrated probabilities
   → Physics-informed features: U-shape vs V-shape index
   → Catalog metadata: contratio, RUWE, stellar params
   → Cross-validation: 5-fold stratified on disposition labels
   → Calibration: Platt scaling for probability scores

2. ACCURACY OF PARAMETER ESTIMATION
   → Physical model: Mandel-Agol transit (batman package)
   → Stellar priors: Teff → limb-darkening coefficients (u1, u2)
   → MCMC sampling: emcee with 100 walkers × 5000 steps
   → Uncertainty: Posterior distributions, ±1σ, ±2σ confidence
   → Validation: Compare with known exoplanet archive parameters

3. METHODS & APPROACH
   → Modular design: 8 independent, testable pipeline stages
   → Reproducible: All random seeds fixed, version-pinned dependencies
   → Documented: Docstrings + Jupyter tutorials + README
   → Validated: Unit tests for each stage; injection-retrieval tests
   → Extensible: Easy to add new classifiers or features

4. VISUALIZATION & CLARITY
   → Phase-folded plots: Raw + binned + model overlay
   → Corner plots: Parameter posterior distributions
   → BLS/TLS periodograms: Power vs period with SDE threshold line
   → Dashboard: Interactive Plotly with zoom/hover/selection
   → Report: LaTeX 3-pager with tables, figures, equations

================================================================================
SECTION 9: REFERENCES
================================================================================

9.1 PRIMARY SOURCES (Directly Used in Pipeline)
------------------------------------------------

[1] Hippke, M. & Heller, R. (2019). "Transit Least Squares: An optimized 
    transit detection algorithm to search for periodic transits of small 
    planets." A&A 623, A39. 
    https://doi.org/10.1051/0004-6361/201834729
    GitHub: https://github.com/hippke/tls

    KEY CONTRIBUTION: TLS achieves 93.1% detection efficiency vs 75.7% for BLS
    at 1% false alarm rate for Earth-sized planets around Sun-like stars.

[2] Kreidberg, L. (2015). "batman: Basic Transit Model cAlculatioN in Python."
    PASP 127, 1161. https://doi.org/10.1086/683602

    KEY CONTRIBUTION: Fast C implementation of Mandel-Agol analytic transit
    model with quadratic limb-darkening.

[3] Foreman-Mackey, D., Hogg, D.W., Lang, D., & Goodman, J. (2013). 
    "emcee: The MCMC Hammer." PASP 125, 306. 
    https://doi.org/10.1086/670067

    KEY CONTRIBUTION: Affine-invariant ensemble MCMC sampler for parameter
    estimation with robust convergence diagnostics.

[4] ExoNet (2026). "Multimodal Deep Learning for TESS Exoplanet Candidate 
    Identification via Phase-Folded Light Curves, Stellar Parameters, and 
    Multi-Head Attention Fusion." arXiv:2604.15560.

    KEY CONTRIBUTION: Trimodal architecture achieving AUC=0.85 on Kepler data,
    identifying 35 high-confidence TESS candidates including habitable-zone
    signals.

[5] Vanderburg, A. & Johnson, J.A. (2014). "A technique for extracting highly 
    precise photometry for the two-wheeled Kepler mission." PASP 126, 948.

    KEY CONTRIBUTION: Systematic detrending methodology for space-based 
    photometry, basis for modern TESS processing.

9.2 SECONDARY SOURCES (Supporting Methods)
-------------------------------------------

[6] Kovács, G., Zucker, S., & Mazeh, T. (2002). "A box-fitting algorithm 
    in the search for periodic transits." A&A 391, 369.

    KEY CONTRIBUTION: Original BLS algorithm — standard baseline for comparison.

[7] Shallue, C.J. & Vanderburg, A. (2018). "Identifying exoplanets with deep 
    learning: A five-planet resonant chain around Kepler-80 and an eighth 
    planet around Kepler-90." AJ 155, 94.

    KEY CONTRIBUTION: First CNN-based exoplanet discovery (AstroNet).

[8] Osborn, H.P. et al. (2020). "Rapid classification of TESS planet 
    candidates with convolutional neural networks." A&A 633, A53.

    KEY CONTRIBUTION: CNN vetting for TESS candidates.

[9] Pearson, K.A., Palafox, L., & Griffith, C.A. (2018). "Searching for 
    exoplanets using artificial intelligence." MNRAS 474, 478.

    KEY CONTRIBUTION: Comparative study showing BLS has lower false negatives
    than ML classifiers alone.

[10] Mandel, K. & Agol, E. (2002). "Analytic light curves for planetary 
     transit searches." ApJ 580, L171.

     KEY CONTRIBUTION: Foundation of analytic transit light curve modeling.

[11] Maxted, P.F.L. (2016). "ellc: A fast, flexible light curve model for 
     detached eclipsing binary stars and transiting exoplanets." A&A 591, A111.

     KEY CONTRIBUTION: General light curve model for binary systems.

[12] Caceres, G.A. et al. (2019). "Transit Comb Filter (TCF) periodogram."

     KEY CONTRIBUTION: Improved periodogram for autocorrelated noise.

[13] Melton et al. (2023). "Autocorrelation in TESS light curves after 
     detrending." ~36% retain significant short-memory autocorrelation.

     KEY CONTRIBUTION: Motivation for ARIMA+TCF or advanced detrending.

[14] Sengupta, S. (2026). "Detection and Characterization of Exoplanet 
     Transits in Kepler Light Curves." NHSJS, February 2026.

     KEY CONTRIBUTION: BLS parameter estimation accuracy <1% for Rₚ/R★.

[15] Frontiers in Astronomy (2026). "Detection of exoplanets from TESS 
     imaging data using unsupervised machine learning techniques." 
     Frontiers in Space Sciences, May 2026.

     KEY CONTRIBUTION: Unsupervised pipeline for TESS exoplanet detection.

9.3 SOFTWARE & DATA RESOURCES
------------------------------

TESS Data Archive: https://archive.stsci.edu/tess/
MAST Portal: https://mast.stsci.edu/
TIC v8.1 Catalog: https://archive.stsci.edu/missions/tess/catalogs/
xCTL v8.01: https://archive.stsci.edu/missions/tess/catalogs/xctl/

Python Packages:
- lightkurve: https://docs.lightkurve.org/
- transitleastsquares: https://github.com/hippke/tls
- batman-package: https://github.com/lkreidberg/batman
- emcee: https://emcee.readthedocs.io/
- wotan: https://github.com/hippke/wotan
- xgboost: https://xgboost.readthedocs.io/
- lightgbm: https://lightgbm.readthedocs.io/

NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
TESS TOI Catalog: https://tess.mit.edu/toi-releases/

================================================================================
                              END OF DOCUMENT
================================================================================

Document prepared for:
  ISRO BAH 2026 Hackathon
  Problem Statement 7: AI-Enabled Detection of Exoplanets from 
  Noisy Astronomical Light Curves

Team: WarpDrive
Team Leader: Omprakash Behera
Institution: Government College of Engineering Kalahandi

Date: June 2026
Version: 1.0
