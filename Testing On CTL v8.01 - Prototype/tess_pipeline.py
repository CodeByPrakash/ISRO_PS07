from manim import *
import numpy as np

# ─────────────────────────────────────────────
#  TESS ML Pipeline — Manim Animation
#  Run: manim -pql tess_pipeline.py TESSDataFlow
#  High quality: manim -pqh tess_pipeline.py TESSDataFlow
# ─────────────────────────────────────────────

BG       = "#050A14"   # deep space black
CYAN     = "#00D4FF"   # TESS teal
PURPLE   = "#A855F7"   # nebula purple
GOLD     = "#F59E0B"   # star gold
RED_CLR  = "#EF4444"   # red dwarf
GREEN    = "#10B981"   # planet green
WHITE_DIM= "#CBD5E1"   # dim white


def make_box(text_lines, color, width=3.2, height=1.3, font_size=15):
    box = RoundedRectangle(
        corner_radius=0.18, width=width, height=height,
        color=color, fill_color=color, fill_opacity=0.12,
        stroke_width=2
    )
    label = Text("\n".join(text_lines), font_size=font_size,
                 color=WHITE, line_spacing=1.1).move_to(box)
    return VGroup(box, label)


class TESSDataFlow(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Starfield ────────────────────────────────────
        stars = VGroup(*[
            Dot(
                point=[np.random.uniform(-7, 7),
                       np.random.uniform(-4, 4), 0],
                radius=np.random.uniform(0.01, 0.03),
                color=WHITE,
                fill_opacity=np.random.uniform(0.3, 0.9)
            )
            for _ in range(140)
        ])
        self.add(stars)

        # ── SCENE 1: Title ───────────────────────────────
        title = Text("TESS ML PIPELINE", font_size=52, color=CYAN,
                     weight=BOLD)
        sub = Text("9.48 Million Stars  ·  3 Classes  ·  1 Mission",
                   font_size=22, color=PURPLE)
        sub.next_to(title, DOWN, buff=0.3)

        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(sub))

        # ── SCENE 2: Data Source ─────────────────────────
        scene_label = Text("STAGE 1 — DATA INGESTION", font_size=20,
                           color=GOLD).to_edge(UP, buff=0.3)
        self.play(FadeIn(scene_label))

        csv_box = make_box(
            ["exo_CTL_08.01.csv", "497 MB  |  9.48M rows",
             "[ID] [priority] [splists] [objID]"],
            GOLD, width=3.8, height=1.4
        ).shift(LEFT * 4.5)

        self.play(Create(csv_box[0]), Write(csv_box[1]), run_time=1)

        # Animated arrow → chunking
        arrow1 = Arrow(csv_box.get_right(), LEFT * 1.5,
                       color=CYAN, buff=0.1, stroke_width=3)
        chunk_info = Text("500K-row\nchunks ×18", font_size=16,
                          color=CYAN).next_to(arrow1, UP, buff=0.1)
        self.play(Create(arrow1), Write(chunk_info))

        # Three class buckets
        classes = [
            ("🪐 PLANET", "5,545,136\n(58.4%)", CYAN,   UP * 1.3),
            ("❄️  COOL DWARF", "3,940,291\n(41.5%)", PURPLE, ORIGIN),
            ("🔥 HOT SUBDWARF", "2,855\n(0.03%)", RED_CLR, DOWN * 1.3),
        ]
        bucket_group = VGroup()
        for name, count, color, offset in classes:
            b = make_box([name, count], color,
                         width=2.8, height=1.0, font_size=14)
            b.shift(RIGHT * 2.5 + offset)
            bucket_group.add(b)
            self.play(Create(b[0]), Write(b[1]), run_time=0.5)

        # Imbalance warning
        warn = Text("⚠  Class Imbalance!", font_size=18,
                    color=RED_CLR).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(warn))
        self.wait(1)
        self.play(FadeOut(scene_label), FadeOut(csv_box),
                  FadeOut(arrow1), FadeOut(chunk_info),
                  FadeOut(bucket_group), FadeOut(warn))

        # ── SCENE 3: Baseline Model ──────────────────────
        scene_label2 = Text("STAGE 2 — BASELINE: RANDOM FOREST",
                            font_size=20, color=GOLD).to_edge(UP, buff=0.3)
        self.play(FadeIn(scene_label2))

        # Growing trees — simple rects representing 200 trees
        trees = VGroup(*[
            Rectangle(
                width=0.12, height=np.random.uniform(0.4, 1.4),
                color=GREEN, fill_color=GREEN, fill_opacity=0.7
            ).shift(LEFT * 5 + RIGHT * i * 0.18 +
                    DOWN * np.random.uniform(-0.3, 0.3))
            for i in range(35)
        ])

        for tree in trees:
            tree.stretch_to_fit_height(0.1)  # start small

        self.play(LaggedStart(*[
            tree.animate.stretch_to_fit_height(
                np.random.uniform(0.5, 1.5)).set_y(
                np.random.uniform(-1.2, 1.2))
            for tree in trees
        ], lag_ratio=0.05, run_time=2))

        rf_label = Text("Random Forest\n200 Trees", font_size=22,
                        color=GREEN).shift(LEFT * 3.2)
        self.play(Write(rf_label))

        # Accuracy gauge
        acc_label = Text("Accuracy:", font_size=20,
                         color=WHITE_DIM).shift(RIGHT * 1.5 + UP * 0.5)
        acc_val = Text("61%", font_size=48,
                       color=GREEN).shift(RIGHT * 1.5 + DOWN * 0.2)
        f1_label = Text("F1-Weighted: 0.57", font_size=18,
                        color=PURPLE).shift(RIGHT * 1.5 + DOWN * 1.2)
        miss = Text("❌ HOT_SUBDWARF missed (F1 = 0.00)",
                    font_size=17, color=RED_CLR).shift(DOWN * 2.2)

        self.play(FadeIn(acc_label), Write(acc_val),
                  FadeIn(f1_label))
        self.play(FadeIn(miss))
        self.wait(1.2)
        self.play(FadeOut(scene_label2), FadeOut(trees),
                  FadeOut(rf_label), FadeOut(acc_label),
                  FadeOut(acc_val), FadeOut(f1_label), FadeOut(miss))

        # ── SCENE 4: XGBoost Model ───────────────────────
        scene_label3 = Text("STAGE 3 — XGBOOST + GPU (CUDA)",
                            font_size=20, color=GOLD).to_edge(UP, buff=0.3)
        self.play(FadeIn(scene_label3))

        bolt = Text("⚡", font_size=80).shift(UP * 0.5)
        self.play(FadeIn(bolt, scale=0.1))
        self.play(bolt.animate.scale(1.5).set_color(GOLD))
        self.play(FadeOut(bolt))

        xgb_box = make_box(
            ["XGBoost Classifier", "300 Trees  |  CUDA GPU",
             "Saves: star_classifier.pkl"],
            CYAN, width=4.5, height=1.6
        )
        self.play(Create(xgb_box[0]), Write(xgb_box[1]))

        acc_xgb = Text("Accuracy: 72%", font_size=36,
                       color=CYAN).shift(DOWN * 1.8)
        f1_xgb = Text("F1-Weighted: 0.72 ✅", font_size=22,
                      color=GREEN).shift(DOWN * 2.7)
        self.play(Write(acc_xgb))
        self.play(FadeIn(f1_xgb))
        self.wait(1.2)
        self.play(FadeOut(scene_label3), FadeOut(xgb_box),
                  FadeOut(acc_xgb), FadeOut(f1_xgb))

        # ── SCENE 5: Feature Engineering ─────────────────
        scene_label4 = Text("STAGE 4 — PHYSICS-INFORMED FEATURES",
                            font_size=20, color=GOLD).to_edge(UP, buff=0.3)
        self.play(FadeIn(scene_label4))

        center = Dot(ORIGIN, color=CYAN, radius=0.15)
        center_label = Text("46\nFeatures", font_size=18,
                            color=CYAN).move_to(ORIGIN)
        self.play(Create(center), Write(center_label))

        feature_groups = [
            ("Astrometry\nra, dec, pmRA…",   GOLD,   UP * 2.2 + LEFT * 3),
            ("Photometry\nTmag, Vmag, Jmag…", PURPLE, UP * 2.2 + RIGHT * 3),
            ("Stellar Physics\nTeff, logg, rad…", GREEN, DOWN * 2.2 + LEFT * 3),
            ("Derived\nJ_K, V_K, Teff_logg…", CYAN,  DOWN * 2.2 + RIGHT * 3),
        ]

        for text, color, pos in feature_groups:
            node = make_box(text.split("\n"), color,
                            width=2.8, height=1.0, font_size=14)
            node.shift(pos)
            line = Line(ORIGIN, pos * 0.75, color=color,
                        stroke_opacity=0.6)
            self.play(Create(line), Create(node[0]),
                      Write(node[1]), run_time=0.6)

        self.wait(1.5)
        self.play(FadeOut(scene_label4), *[FadeOut(m) for m in self.mobjects
                                           if m is not stars])

        # ── SCENE 6: Classification Output ───────────────
        scene_label5 = Text("OUTPUT — CLASSIFICATION RESULTS",
                            font_size=20, color=GOLD).to_edge(UP, buff=0.3)
        self.play(FadeIn(scene_label5))

        ensemble = make_box(
            ["Ensemble Model", "Random Forest + Gradient Boosting",
             "Soft Voting  |  5-Fold CV"],
            PURPLE, width=5, height=1.5
        ).shift(LEFT * 2.5)
        self.play(Create(ensemble[0]), Write(ensemble[1]))

        out_classes = [
            ("🪐  Planet Candidates", CYAN,   UP * 1.2 + RIGHT * 3.2),
            ("❄️   Cool Dwarf Stars",  PURPLE, ORIGIN   + RIGHT * 3.2),
            ("🔥  Hot Subdwarfs",      RED_CLR, DOWN * 1.2 + RIGHT * 3.2),
        ]
        for label, color, pos in out_classes:
            node = make_box([label], color,
                            width=3, height=0.75, font_size=16)
            node.shift(pos)
            arrow = Arrow(ensemble.get_right(), node.get_left(),
                          color=color, buff=0.08, stroke_width=2)
            self.play(Create(arrow), Create(node[0]),
                      Write(node[1]), run_time=0.5)

        f1_final = Text("F1-Macro: ~0.85 ± 0.02", font_size=24,
                        color=GOLD).to_edge(DOWN, buff=0.6)
        self.play(Write(f1_final))
        self.wait(2)

        # ── SCENE 7: Outro ────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects if m is not stars])
        outro = Text('"Every classified star is a potential new world."',
                     font_size=26, color=CYAN, slant=ITALIC)
        credit = Text("Hack2Skill × ISRO  |  TESS xCTL v8.01",
                      font_size=18, color=PURPLE)
        credit.next_to(outro, DOWN, buff=0.4)
        self.play(FadeIn(outro), FadeIn(credit))
        self.wait(2.5)
        self.play(FadeOut(outro), FadeOut(credit), FadeOut(stars))


# ─────────────────────────────────────────────
#  BONUS: Quick Class Distribution Bar Chart
#  Run: manim -pql tess_pipeline.py ClassDistribution
# ─────────────────────────────────────────────

class ClassDistribution(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("TESS CTL v8.01 — Class Distribution",
                     font_size=34, color=CYAN).to_edge(UP, buff=0.4)
        self.play(Write(title))

        data = [
            ("planetcandidate",  5545136, CYAN),
            ("cooldwarfs_v8",    3940291, PURPLE),
            ("hotsubdwarfs_v8",  2855,    RED_CLR),
        ]
        max_val = 5545136
        bar_max_h = 3.5

        bars = VGroup()
        for i, (label, count, color) in enumerate(data):
            h = (count / max_val) * bar_max_h
            bar = Rectangle(width=1.4, height=h, color=color,
                            fill_color=color, fill_opacity=0.8)
            bar.move_to([i * 2.6 - 2.6, -2 + h / 2, 0])

            pct = f"{count / 9488282 * 100:.2f}%"
            bar_label = Text(label.replace("_v8", ""),
                             font_size=13, color=WHITE)
            bar_label.next_to(bar, DOWN, buff=0.15)
            count_label = Text(f"{count:,}", font_size=14, color=color)
            count_label.next_to(bar, UP, buff=0.1)
            pct_label = Text(pct, font_size=13, color=GOLD)
            pct_label.next_to(count_label, UP, buff=0.05)

            self.play(
                bar.animate.set_height(h).move_to(
                    [i * 2.6 - 2.6, -2 + h / 2, 0]),
                FadeIn(bar_label), FadeIn(count_label),
                FadeIn(pct_label), run_time=1.2
            )
            bars.add(bar)

        warn = Text("⚠  hotsubdwarfs: severely imbalanced (0.03%)",
                    font_size=18, color=RED_CLR).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(warn))
        self.wait(2)
