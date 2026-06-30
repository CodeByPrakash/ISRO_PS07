# 🛸 TESS ML WORKFLOW — VISUAL GUIDE & ANIMATION ROADMAP

> **Theme**: Deep Space / Cosmic Intelligence  
> **Project**: TESS Exoplanet Classification | Hack2Skill × ISRO  
> **Purpose**: Visualizing the end-to-end ML pipeline as an animated presentation

---

## 🎨 VISUAL THEME SPECIFICATION

```
╔══════════════════════════════════════════════════════════════╗
║   THEME: "STELLAR INTELLIGENCE"                              ║
║   ─────────────────────────────────────────────────         ║
║   Background  →  Deep space black  (#050A14)                 ║
║   Primary     →  TESS teal/cyan    (#00D4FF)                 ║
║   Accent 1    →  Nebula purple     (#A855F7)                 ║
║   Accent 2    →  Star gold         (#F59E0B)                 ║
║   Danger      →  Red dwarf         (#EF4444)                 ║
║   Success     →  Planet green      (#10B981)                 ║
║   Font        →  Orbitron (headers) + Inter (body)          ║
║   Particles   →  Animated star field (CSS/JS)               ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 🗺️ THE FULL ML PIPELINE — VISUAL FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🌌 TESS ML INTELLIGENCE PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

        ╔══════════════╗       STREAMS IN 500K-ROW CHUNKS
        ║  🛰️ RAW DATA  ║ ──────────────────────────────────────────────────►
        ║  exo_CTL     ║       (497 MB, 9.48M rows, 4 columns)
        ║  v8.01.csv   ║
        ╚══════════════╝
                │
                ▼
        ┌──────────────────────────────┐
        │  📊 STAGE 1: DATA INGESTION  │  ← Animated: scrolling data stream
        │  ─────────────────────────── │
        │  • Chunk-based CSV streaming │
        │  • Category count tracking   │
        │  • Stratified 1% sampling    │
        │  • Output: training_sample   │
        └──────────────┬───────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌───────────┐ ┌──────────┐ ┌────────────┐
    │🪐 PLANET  │ │❄️ COOL   │ │🔥 HOT SUB  │
    │5,545,136  │ │ DWARF    │ │  DWARF     │
    │  (58.4%)  │ │3,940,291 │ │   2,855    │
    │           │ │ (41.5%)  │ │  (0.03%)   │
    └───────────┘ └──────────┘ └────────────┘
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  🔬 STAGE 2: BASELINE MODEL  │  ← Animated: decision tree growing
        │  ─────────────────────────── │
        │  Random Forest (200 trees)   │
        │  • Feature: priority only    │
        │  • Accuracy: 61%             │
        │  • F1: 0.57 (weighted)       │
        │  • ❌ HOT_SUBDWARF missed    │
        └──────────────┬───────────────┘
                       │
                       ▼  [IMPROVE →]
        ┌──────────────────────────────┐
        │  ⚡ STAGE 3: XGBOOST MODEL   │  ← Animated: GPU lightning bolts
        │  ─────────────────────────── │
        │  XGBoost (300 trees, CUDA)   │
        │  • Accuracy: 72%             │
        │  • F1: 0.72 (weighted)       │
        │  • Saves: star_classifier    │
        │  • Saves: label_encoder      │
        └──────────────┬───────────────┘
                       │
                       ▼
        ╔══════════════════════════════╗
        ║  🧬 STAGE 4: FULL PROTOTYPE  ║  ← Animated: neural web forming
        ║  ─────────────────────────── ║
        ║  Physics-Informed Features   ║
        ║  ┌─────────┬──────────────┐  ║
        ║  │ 36 BASE │ 10 ENGINEERED│  ║
        ║  │FEATURES │  FEATURES    │  ║
        ║  └─────────┴──────────────┘  ║
        ║  ENSEMBLE: RF + GB (voting)  ║
        ║  5-Fold CV F1-Macro: ~0.85   ║
        ╚══════════════════════════════╝
                       │
                       ▼
        ┌──────────────────────────────┐
        │  🎯 OUTPUT: CLASSIFICATION   │  ← Animated: stars sorting by class
        │  ─────────────────────────── │
        │  🪐 Planet Candidates        │
        │  ❄️  Cool Dwarf Stars         │
        │  🔥 Hot Subdwarf Stars       │
        │  → predictions.csv           │
        └──────────────────────────────┘
```

---

## 🎬 ANIMATION & PRESENTATION TOOL OPTIONS

### 🏆 OPTION 1 — Manim (Python — Best for Data Science)
**Best for**: Precise, mathematical, cinematic quality animations  
**Used by**: 3Blue1Brown (YouTube)

```bash
pip install manim
```

**What you'd animate**:
- Data flowing as glowing particles from a CSV
- Decision trees growing branch by branch
- Confusion matrix heatmap building cell-by-cell
- Feature importance bar chart rising with glow effects
- Stars being sorted into class bins by laser beams

**Manim Scene Concept**:
```python
# Pseudo-code for your pipeline animation
class TESSPipeline(Scene):
    def construct(self):
        # Scene 1: Raw data stream
        data_stream = DataStream("497MB CSV")
        self.play(data_stream.stream_animation())  # particles flowing

        # Scene 2: Chunked processing
        chunks = ChunkBlocks(n=18, label="500K rows each")
        self.play(Create(chunks))

        # Scene 3: Model training
        forest = RandomForestVisual(n_trees=200)
        self.play(forest.grow_trees())

        # Scene 4: Accuracy metrics
        metrics = AccuracyMeter(value=0.72)
        self.play(metrics.fill_to_value())
```

---

### 🥈 OPTION 2 — Flourish / Datawrapper (No-Code, Browser)
**Best for**: Quick, shareable, interactive charts  
**URL**: https://flourish.studio | https://www.datawrapper.de

**What to create**:
- Animated bar race: class distribution across chunks
- Sankey diagram: raw data → sample → train/test → predictions
- Bubble chart: feature importance vs. physical meaning

---

### 🥉 OPTION 3 — Canva / PowerPoint Morph (Presentation)
**Best for**: Hackathon presentations, judges panel  

**Slide Sequence**:
```
Slide 1: TESS satellite + starfield + title (fade in)
Slide 2: 497MB CSV → animated data river (morph)
Slide 3: 3 class bubbles floating and colliding (morph)
Slide 4: Decision tree growing (animation)
Slide 5: Accuracy gauge filling: 61% → 72% → 85% (morph)
Slide 6: Feature engineering web exploding outward (zoom)
Slide 7: Final classification: stars sorted by type (morph)
Slide 8: Results + future work (fade out to starfield)
```

---

### 🌐 OPTION 4 — Interactive Web App (HTML/CSS/JS)
**Best for**: Live demo during hackathon judging, browser-based  

**Tech Stack**:
- **D3.js** — force-directed graph of feature relationships
- **Three.js** — 3D star field with color-coded classification
- **Anime.js** — smooth pipeline flow animations
- **Chart.js** — live confusion matrix + metric gauges

---

### 🤖 OPTION 5 — Streamlit Dashboard (Python — Fastest)
**Best for**: Quick interactive demo, runs locally  

```bash
pip install streamlit plotly
streamlit run dashboard.py
```

**Components**:
- Pipeline progress bar (stage 1 → 4)
- Live class distribution donut chart
- Interactive feature importance slider
- Confusion matrix heatmap
- Prediction result table with color coding

---

## 📐 RECOMMENDED ANIMATION STORYBOARD

### ACT 1 — THE UNIVERSE OF DATA (0:00 - 0:30)
```
🌌 Black screen
   ↓ Stars appear one by one (particle burst)
   ↓ TESS satellite sweeps across
   ↓ Title: "TESS EXOPLANET CLASSIFIER" materializes
   ↓ Subtitle: "9.48 Million Stars. 3 Classes. 1 Mission."
```

### ACT 2 — DATA INGESTION (0:30 - 1:00)
```
📊 497MB CSV file icon glows
   ↓ Green data stream splits into 18 chunks
   ↓ Each chunk: 500,000 rows (counter ticking)
   ↓ 3 buckets fill: 🪐 58.4% | ❄️ 41.5% | 🔥 0.03%
   ↓ HOT_SUBDWARF bucket glows red — IMBALANCE ALERT!
```

### ACT 3 — MODEL TRAINING (1:00 - 1:45)
```
🌲 Random Forest: 200 trees grow simultaneously
   ↓ Accuracy meter: fills to 61% (slows at end)
   ↓ ❌ Cross appears on HOT_SUBDWARF: F1=0.00
   ↓ [UPGRADE] bolt strikes → XGBoost loads
   ↓ GPU fire particles → 300 trees + CUDA
   ↓ Accuracy meter: climbs to 72% ✅
```

### ACT 4 — FEATURE ENGINEERING (1:45 - 2:15)
```
🧬 46 features explode outward from center node
   ↓ Groups cluster: Astrometry | Photometry | Physics
   ↓ Derived features glow brighter (J_K, V_K, Teff_logg)
   ↓ Physics formulas animate: Stefan-Boltzmann law
   ↓ Feature importance bars rise with sparkle effects
```

### ACT 5 — CLASSIFICATION & RESULTS (2:15 - 2:45)
```
⭐ Stars stream in from left (mixed colors)
   ↓ Ensemble model: two beams (RF + GB) merge
   ↓ Stars split into 3 streams by color:
      🔵 Blue  → Planet Candidates
      🔴 Red   → Cool Dwarfs
      🟡 Gold  → Hot Subdwarfs
   ↓ Final scoreboard: F1-Macro 0.85 ± 0.02
   ↓ predictions.csv saves (file write animation)
```

### ACT 6 — OUTRO (2:45 - 3:00)
```
🌠 Zoom out to galaxy view
   ↓ Text: "Every classified star is a potential new world"
   ↓ Hack2Skill × ISRO logo
   ↓ Fade to black with star twinkle
```

---

## 🛠️ QUICK START: MANIM PIPELINE SCENE

Save as `tess_pipeline.py` and run:
```bash
manim -pql tess_pipeline.py TESSDataFlow
```

```python
from manim import *
import numpy as np

class TESSDataFlow(Scene):
    def construct(self):
        # ── Background ──────────────────────────────────
        self.camera.background_color = "#050A14"

        # ── Title ───────────────────────────────────────
        title = Text("TESS ML PIPELINE", font_size=48,
                     color="#00D4FF").to_edge(UP)
        subtitle = Text("9.48M Stars → 3 Classes", font_size=24,
                        color="#A855F7").next_to(title, DOWN)
        self.play(Write(title), FadeIn(subtitle, shift=UP))
        self.wait(0.5)

        # ── Data Source Node ─────────────────────────────
        csv_box = RoundedRectangle(corner_radius=0.2, width=3, height=1.2,
                                   color="#F59E0B").shift(LEFT * 4)
        csv_label = Text("exo_CTL_08.01.csv\n497 MB | 9.48M rows",
                         font_size=16, color=WHITE).move_to(csv_box)
        self.play(Create(csv_box), Write(csv_label))

        # ── Arrow + Chunking ─────────────────────────────
        arrow = Arrow(csv_box.get_right(), ORIGIN + LEFT * 1,
                      color="#00D4FF", buff=0.1)
        self.play(Create(arrow))

        chunk_label = Text("500K-row chunks", font_size=18,
                           color="#00D4FF").next_to(arrow, UP, buff=0.1)
        self.play(Write(chunk_label))

        # ── Model Box ────────────────────────────────────
        model_box = RoundedRectangle(corner_radius=0.2, width=3.5, height=1.5,
                                     color="#10B981").shift(RIGHT * 1.5)
        model_text = Text("XGBoost Classifier\n300 trees | CUDA GPU\nAcc: 72%",
                          font_size=15, color=WHITE).move_to(model_box)
        self.play(Create(model_box), Write(model_text))

        # ── Output Nodes ─────────────────────────────────
        classes = [("🪐 PLANET", "#00D4FF", UP * 1.5),
                   ("❄️ COOL DWARF", "#A855F7", ORIGIN),
                   ("🔥 HOT SUBDWARF", "#EF4444", DOWN * 1.5)]

        for label, color, pos in classes:
            out_box = RoundedRectangle(corner_radius=0.15, width=2.2,
                                       height=0.6, color=color).shift(RIGHT * 4.5 + pos)
            out_text = Text(label, font_size=14, color=WHITE).move_to(out_box)
            out_arrow = Arrow(model_box.get_right(),
                              out_box.get_left(), color=color, buff=0.05)
            self.play(Create(out_box), Write(out_text),
                      Create(out_arrow), run_time=0.6)

        self.wait(2)
```

---

## 📦 TOOL COMPARISON TABLE

| Tool | Difficulty | Quality | Interactive | Best For |
|------|-----------|---------|-------------|----------|
| **Manim** | ⭐⭐⭐ Hard | 🎬 Cinematic | ❌ No | YouTube/Video export |
| **Flourish** | ⭐ Easy | 📊 Good | ✅ Yes | Quick shareable charts |
| **Canva Morph** | ⭐ Easy | 🎨 Good | ❌ Limited | Hackathon slides |
| **D3.js** | ⭐⭐⭐⭐ Expert | 💎 Premium | ✅ Full | Live web demo |
| **Streamlit** | ⭐⭐ Medium | 📊 Functional | ✅ Yes | Python dashboard |
| **PowerPoint** | ⭐ Easy | 🎨 OK | ❌ No | Judge presentations |

---

## 🎯 MY RECOMMENDATION FOR THIS PROJECT

> For a **Hackathon judging panel**, use this stack:

```
1. CANVA (or Google Slides) → Morph transition slides
   └── Use storyboard above as your slide sequence
   └── Add starfield background + Orbitron font
   └── Export as MP4 video (built-in)

2. STREAMLIT → Live interactive demo
   └── Run locally during presentation
   └── Show real confusion matrix from your model
   └── Let judges filter by class

3. MANIM → Pre-render the pipeline animation
   └── Embed as a video in slide 3-5
   └── Shows technical depth to judges
```

---

## 🔗 RESOURCES

| Resource | Link | Purpose |
|----------|------|---------|
| Manim Community | https://www.manim.community | Animation library docs |
| Manim Tutorial | https://docs.manim.community/en/stable/tutorials | Getting started |
| Flourish Studio | https://flourish.studio | No-code animated charts |
| D3.js Gallery | https://observablehq.com/@d3/gallery | Interactive chart examples |
| Streamlit Docs | https://docs.streamlit.io | Dashboard framework |
| Orbitron Font | https://fonts.google.com/specimen/Orbitron | Space-themed typography |
| Three.js Stars | https://threejs.org/examples/#webgl_points_sprites | 3D starfield effect |

---

*🌠 "Visualize the cosmos of data — let every algorithm tell a story among the stars."*

---
**Project**: TESS Exoplanet Classification | xCTL v8.01  
**Theme**: Stellar Intelligence — Deep Space Palette  
**Author**: Hack2Skill × ISRO Hackathon Team
