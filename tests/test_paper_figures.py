from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def _load_paper_figures_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "paper_figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_paper_figures_render(tmp_path: Path) -> None:
    figures = _load_paper_figures_module()

    figures.save_forward_equivalence(tmp_path)
    figures.save_order_hierarchy(tmp_path)
    figures.save_census(tmp_path)
    figures.save_geometry_time_map(tmp_path)

    stems = [
        "figure1_forward_equivalence",
        "figure2_order_hierarchy",
        "figure3_forward_class_census",
        "figure4_order_to_time_geometry",
    ]
    for stem in stems:
        for suffix in ("pdf", "png"):
            path = tmp_path / f"{stem}.{suffix}"
            assert path.exists()
            assert path.stat().st_size > 0
