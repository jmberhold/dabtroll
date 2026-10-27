"""The publication bundle must remain complete, readable, and traceable to its CSV."""

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

from scripts import paper_bundle as bundle
from scripts import paper_results as paper
from tests.test_paper_results import fixture
from tests.test_node_completion_review import episode


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"href", "src"}:
                self.paths.append(value)


def prompt_fixture():
    episodes, nodes, observations = fixture()
    inherited = observations.assign(model_name="qwen35_qwen3_prompt")
    tailored = observations.assign(model_name="qwen35")
    return paper.build_paper_results(episodes, nodes, pd.concat([observations, inherited, tailored]))


class PaperBundleTests(unittest.TestCase):
    def test_registry_covers_every_table(self):
        self.assertEqual(set(bundle.TABLE_INFO), set(paper.TABLE_STAGES))
        labels = [info[0] for info in [*bundle.TABLE_INFO.values(), *bundle.COMPACT_INFO.values(), *bundle.FIGURE_INFO.values()]]
        self.assertEqual(len(labels), len(set(labels)))

    def test_bundle_copies_full_csv_catalogs_all_sections_and_flags_stale_sources(self):
        results = prompt_fixture()
        empty_cases = paper.representative_cases(results).iloc[:0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            # An extra column proves that the source copy is not results.episodes reserialized.
            results.episodes.assign(extra_source_field="preserve all columns").to_csv(source, index=False)
            original = source.read_bytes()
            output = root / "bundle"
            frames = pd.DataFrame({"frame_step": [0, 11, 23], "machine_time_s": [0, 4.4, 9.2]})
            figure, axes = plt.subplots(1, 3)
            for axis in axes:
                axis.imshow(np.zeros((8, 8, 3)))
                axis.axis("off")
            with patch("pandas.read_excel", side_effect=AssertionError("Raw workbook read")), \
                    patch.object(paper, "plot_episode_triptych", return_value=(figure, frames)):
                bundle.export_stage(results, source, output, "01")
            self.assertIn("Incomplete or stale", (output / "index.html").read_text())
            bundle.export_stage(results, source, output, "02")
            with patch.object(paper, "representative_cases", return_value=empty_cases):
                bundle.export_stage(results, source, output, "03")
            catalog = json.loads((output / "manifest.json").read_text())
            self.assertEqual((output / "data/episodes.csv").read_bytes(), original)
            self.assertEqual(catalog["source"]["sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual({e["stage"] for e in catalog["entries"]}, {"01", "02", "03"})
            self.assertEqual(sum(e["kind"] == "figure" for e in catalog["entries"]), 10)
            self.assertEqual(sum("tex" in e["files"] for e in catalog["entries"]), 5)
            self.assertEqual(len({e["label"] for e in catalog["entries"]}), len(catalog["entries"]))
            for entry in catalog["entries"]:
                self.assertTrue(entry["title"] and entry["description"])
                self.assertEqual(entry["source_sha256"], catalog["source"]["sha256"])
                for path in entry["files"].values():
                    self.assertTrue((output / path).is_file(), path)
                if entry["kind"] == "figure":
                    with Image.open(output / entry["files"]["png"]) as image:
                        self.assertEqual(image.width, 2160)
                        self.assertEqual(image.info["Title"], entry["title"])
                    self.assertIn(b"/FontFile2", (output / entry["files"]["pdf"]).read_bytes())
                    if entry["side_by_side"]:
                        with Image.open(output / entry["files"]["side_png"]) as image:
                            self.assertEqual(image.width, 4320)
            parser = Links()
            parser.feed((output / "index.html").read_text())
            self.assertTrue(all((output / path).exists() for path in parser.paths))
            self.assertIn("All three stages exported", (output / "index.html").read_text())
            self.assertIn("<h3>Auditor sensitivity</h3>", (output / "index.html").read_text())
            count = len(catalog["entries"])
            retired = output / "figures/retired.png"
            retired.write_bytes(b"old managed figure")
            catalog["entries"].append({
                **catalog["entries"][0], "stage": "02", "label": "retired",
                "files": {"png": "figures/retired.png"},
            })
            (output / "manifest.json").write_text(json.dumps(catalog))
            figures = {
                "sensitivity": paper.plot_sensitivity(results),
                "prompt": paper.plot_prompt_sensitivity(results),
                "family": paper.plot_node_family_performance(results),
            }
            bundle.export_stage(results, source, output, "02", figures=figures)
            self.assertFalse(retired.exists())
            self.assertEqual(figures["sensitivity"]._suptitle.get_text(), "Auditor sensitivity")
            bundle.export_stage(results, source, output, "02", figures=figures)
            self.assertEqual(len(json.loads((output / "manifest.json").read_text())["entries"]), count)
            source.write_bytes(original + b"\n")
            bundle.export_stage(results, source, output, "02")
            self.assertIn("STALE", (output / "index.html").read_text())
            self.assertEqual((output / "data/episodes.csv").read_bytes(), original + b"\n")
        plt.close("all")

    def test_publication_fonts_bounds_and_nonoverlapping_tick_labels(self):
        results = prompt_fixture()
        for name, factory in (
            ("sensitivity", paper.plot_sensitivity),
            ("prompt", paper.plot_prompt_sensitivity),
            ("family", paper.plot_node_family_performance),
            ("sequential", paper.plot_sequential_completion),
        ):
            with self.subTest(name=name), plt.rc_context(bundle.STYLE):
                figure = factory(results)
                bundle.prepare_figure(figure, name, "Publication figure")
                self.assertAlmostEqual(figure.get_figwidth(), 7.2)
                for text in figure.findobj(Text):
                    self.assertGreaterEqual(text.get_fontsize(), 9.5)
                renderer = figure.canvas.get_renderer()
                bounds = figure.get_tightbbox(renderer)
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertLessEqual(bounds.x1, 7.2)
                self.assertGreaterEqual(bounds.y0, 0)
                self.assertLessEqual(bounds.y1, figure.get_figheight())
                for axis in figure.axes:
                    labels = [text for text in axis.get_xticklabels() if text.get_visible() and text.get_text()]
                    for left, right in zip(labels, labels[1:]):
                        self.assertFalse(left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer)))
                plt.close(figure)

    def test_exported_case_gutters_match_image_border(self):
        figure, axes = plt.subplots(1, 3)
        for axis in axes:
            axis.imshow(np.zeros((8, 12, 3)))
            axis.set_title("frame_00010.jpg\nresponse evidence")
            axis.axis("off")
        bundle.prepare_figure(figure, "case", "Luna 6: Missed complete")
        boxes = [axis.get_position() for axis in figure.axes]
        self.assertAlmostEqual(figure.get_figwidth(), 7.2)
        self.assertAlmostEqual(boxes[0].x0, 0.008)
        self.assertAlmostEqual(1 - boxes[-1].x1, 0.008)
        for left, right in zip(boxes, boxes[1:]):
            self.assertAlmostEqual(right.x0 - left.x1, 0.008)
        bounds = figure.get_tightbbox(figure.canvas.get_renderer())
        self.assertGreaterEqual(bounds.x0, 0)
        self.assertGreaterEqual(bounds.y0, 0)
        self.assertLessEqual(bounds.x1, 7.2)
        self.assertLessEqual(bounds.y1, figure.get_figheight())
        plt.close(figure)

    def test_small_latex_tables_have_captions_labels_and_escaped_content(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            frame = pd.DataFrame({"Model name": ["Qwen_3.8 & <review>"], "Percent": ["90%"]})
            entry = bundle._write_table(output, "03", "t99_test", "Title & result",
                                        "Description with 90% agreement.", frame, compact=True)
            latex = (output / entry["files"]["tex"]).read_text()
            for token in (r"\caption{", r"\label{tab:vibtarve_t99_test}", r"\toprule", r"\shortstack{Model\\name}",
                          r"Qwen\_3.8 \&", r"90\%", r"\small"):
                self.assertIn(token, latex)
            self.assertNotIn("resizebox", latex)
            self.assertNotIn("scriptsize", latex)
            html = (output / entry["files"]["html"]).read_text()
            self.assertIn("&lt;review&gt;", html)
            self.assertIn("<title>Title &amp; result</title>", html)
            self.assertIn("<h1>Title &amp; result</h1>", html)
            self.assertEqual(pd.read_csv(output / entry["files"]["csv"]).iloc[0]["Percent"], "90%")

    def test_side_by_side_layout_preserves_paper_font_size_and_panel_order(self):
        results = prompt_fixture()
        for name, factory in (
            ("corpus", paper.plot_corpus), ("endpoint", paper.plot_simulator_endpoint),
            ("prompt", paper.plot_prompt_sensitivity), ("family", paper.plot_node_family_performance),
            ("sequential", paper.plot_sequential_completion), ("timing", paper.plot_timing),
        ):
            with self.subTest(name=name), plt.rc_context(bundle.STYLE):
                figure = factory(results)
                bundle.prepare_figure(figure, name, "Side-by-side paper figure", layout="side_by_side")
                self.assertAlmostEqual(figure.get_figwidth(), 14.4)
                self.assertTrue(all(text.get_fontsize() >= 19 for text in figure.findobj(Text)))
                axes = figure.axes[:3 if name == "sequential" else 2]
                for left, right in zip(axes, axes[1:]):
                    self.assertLess(left.get_position().x1, right.get_position().x0)
                if name == "timing":
                    renderer = figure.canvas.get_renderer()
                    for axis in axes:
                        labels = axis.get_xticklabels()
                        for label in labels:
                            self.assertFalse(label.get_window_extent(renderer).overlaps(
                                figure.legends[0].get_window_extent(renderer)))
                        for left, right in zip(labels, labels[1:]):
                            self.assertFalse(left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer)))
                plt.close(figure)

    def test_five_revision_supplement_is_development_only_and_csv_backed(self):
        rows = []
        for index in range(2):
            name = f"Task_episode_1_of_21_run{index}"
            row = {}
            for version in range(1, 6):
                row.update(episode(name, ["0:00", "", "0:02"], ["0:00", "0:01", ""], version))
                row.update(episode(name, ["0:00", "", "0:02"], ["", "", ""], version, family="luna6"))
            row["molmo_observation_count"] = 3
            rows.append(row)
        data = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory, \
                patch("pandas.read_excel", side_effect=AssertionError("Raw workbook read")):
            output = Path(directory) / "prompt_development"
            figure, review, index = bundle.export_prompt_development(data, output)
            self.assertEqual(review["counts"]["model"].drop_duplicates().tolist(),
                             [f"Qwen3.8 v{i}" for i in range(1, 6)])
            self.assertEqual([t.get_text() for t in figure.axes[0].get_yticklabels()],
                             [f"v{i}" for i in range(1, 6)])
            self.assertTrue(review["counts"][list(paper.review.COMPLETION_CATEGORIES)].sum(axis=1).eq(review["counts"].eligible_count).all())
            catalog = json.loads((output / "manifest.json").read_text())
            self.assertEqual(catalog["scope"], "prompt_development_only")
            self.assertEqual(set(catalog["families"]), {"qwen38", "luna6"})
            self.assertEqual(len(catalog["entries"]), 8)
            self.assertIn("Luna 6 v1-v5", index.read_text())
            luna = pd.read_csv(output / "tables/p07_luna6_counts.csv")
            self.assertEqual(luna.model.drop_duplicates().tolist(), [f"Luna 6 v{i}" for i in range(1, 6)])
            self.assertTrue(luna[list(paper.review.COMPLETION_CATEGORIES)].sum(axis=1).eq(luna.eligible_count).all())
            self.assertTrue((output / "tables/p03_counts.tex").is_file())
            self.assertTrue((output / "tables/p07_luna6_counts.tex").is_file())
            self.assertIn("Prompt", (output / "tables/p07_luna6_counts.tex").read_text())
            self.assertNotIn("eligible\\_count", (output / "tables/p07_luna6_counts.tex").read_text())
            parser = Links()
            parser.feed(index.read_text())
            self.assertTrue(all((output / path).exists() for path in parser.paths))
            self.assertIn("not held-out evaluation", index.read_text())
            self.assertEqual(len(pd.read_csv(output / "data/episodes.csv")), 2)
            for path in catalog["entries"][0]["files"].values():
                self.assertTrue((output / path).is_file())
            with self.assertRaisesRegex(ValueError, "All five"):
                bundle.export_prompt_development(data.drop(columns="qwen38_v3_observation_count"), output)
            with self.assertRaisesRegex(ValueError, "All five Luna 6"):
                bundle.export_prompt_development(data.drop(columns="luna6_v5_observation_count"), output)
            with self.assertRaisesRegex(ValueError, "every episode"):
                bundle.export_prompt_development(data.assign(luna6_v4_observation_count=[3, 0]), output)
            with self.assertRaisesRegex(ValueError, "episode-1"):
                bundle.export_prompt_development(data.assign(episode_dir="Task_episode_2_of_21_run"), output)
            plt.close(figure)

    def test_development_header_gap_and_journal_fonts_for_both_families(self):
        data = pd.DataFrame([{
            **episode("Task_episode_1_of_21_run", ["0:00", "", ""], ["0:00", "", ""]),
            **episode("Task_episode_1_of_21_run", ["0:00", "", ""], ["0:00", "", ""], family="luna6"),
        }])
        for family, label in (("qwen38", "Qwen3.8"), ("luna6", "Luna 6")):
            review = paper.review.build_node_completion_review(
                data, include_reference_models=False,
                qwen38_versions=[1] if family == "qwen38" else [],
                luna_versions=[1] if family == "luna6" else [],
            )
            for layout, scale in (("stacked", 1), ("side_by_side", 2)):
                with self.subTest(family=family, layout=layout), plt.rc_context(bundle.STYLE):
                    figure, _ = paper.review.plot_node_completion_review(review["counts"])
                    bundle.prepare_figure(
                        figure, "sequential", f"{label} prompt development: v1--v5",
                        layout=layout, compact_header=True,
                    )
                    renderer = figure.canvas.get_renderer()
                    heading = figure._suptitle.get_window_extent(renderer)
                    panel_top = max(axis.title.get_window_extent(renderer).y1 for axis in figure.axes)
                    gap_in = (heading.y0 - panel_top) / figure.dpi / scale
                    self.assertGreaterEqual(gap_in, 0.09)
                    self.assertLessEqual(gap_in, 0.11)
                    self.assertTrue(all(text.get_fontsize() / scale >= 9.5 for text in figure.findobj(Text)))
                    bounds = figure.get_tightbbox(renderer)
                    self.assertGreaterEqual(bounds.y0, 0)
                    self.assertLessEqual(bounds.y1, figure.get_figheight())
                    plt.close(figure)

    def test_invalid_stage_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Unknown results stage"):
                bundle.export_stage(prompt_fixture(), "missing.csv", Path(directory) / "output", "99")
            self.assertFalse((Path(directory) / "output").exists())


if __name__ == "__main__":
    unittest.main()
