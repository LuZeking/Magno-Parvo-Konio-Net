import importlib.util
import fnmatch
import sys
import types
import unittest
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]


def _ensure_package(name: str, path: Optional[Path] = None) -> None:
    if name in sys.modules:
        return

    module = types.ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeSeries:
    def __init__(self, values):
        self._values = list(values)

    def unique(self):
        return list(self._values)


class _FakeDataFrame:
    def __init__(self, subjects):
        self.subj = _FakeSeries(subjects)


def _install_simple_eval_stubs():
    numpy = types.ModuleType("numpy")
    numpy.array = lambda x: x
    sys.modules["numpy"] = numpy

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = object
    sys.modules["pandas"] = pandas

    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    sys.modules["torch"] = torch

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda iterable, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_mod

    _ensure_package("neuroai", ROOT / "neuroai")
    _ensure_package("neuroai.evaluation", ROOT / "neuroai" / "evaluation")
    _ensure_package("neuroai.plotting", ROOT / "neuroai" / "plotting")

    _load_module("neuroai.constants", ROOT / "neuroai" / "constants.py")

    evaluation_mod = types.ModuleType("neuroai.evaluation.evaluate")
    evaluation_mod.ResultPrinter = type("ResultPrinter", (), {})
    evaluation_mod.print_performance_to_csv = lambda *args, **kwargs: None
    sys.modules["neuroai.evaluation.evaluate"] = evaluation_mod

    analyses_mod = types.ModuleType("neuroai.plotting.analyses")

    class ShapeBias:
        def analysis(self, df=None):
            return {
                "shape-bias": 0.0,
                "fraction-correct-shape": 0.0,
                "fraction-correct-texture": 0.0,
            }

    analyses_mod.ShapeBias = ShapeBias
    sys.modules["neuroai.plotting.analyses"] = analyses_mod

    repr_mod = types.ModuleType("neuroai.repr_shape_bias")
    repr_mod.compute_repr_shape_bias = lambda *args, **kwargs: {}
    sys.modules["neuroai.repr_shape_bias"] = repr_mod

    utils_mod = types.ModuleType("neuroai.utils")
    utils_mod.load_dataset = lambda *args, **kwargs: None
    utils_mod.load_model = lambda *args, **kwargs: (None, "pytorch")
    sys.modules["neuroai.utils"] = utils_mod

    colors_mod = types.ModuleType("neuroai.plotting.colors")
    colors_mod.rgb = lambda r, g, b, divide_by=255.0: (r / divide_by, g / divide_by, b / divide_by)
    sys.modules["neuroai.plotting.colors"] = colors_mod

    decision_makers_mod = types.ModuleType("neuroai.plotting.decision_makers")

    def is_human_decision_maker(name):
        return (
            fnmatch.fnmatch(name, "subject-*")
            or name == "adults"
            or fnmatch.fnmatch(name, "*_years_old")
        )

    class DecisionMaker:
        def __init__(self, name_pattern, df, color="grey", marker="o", plotting_name=None):
            patterns = [name_pattern] if isinstance(name_pattern, str) else list(name_pattern)
            self.name_pattern = patterns
            self.color = color
            self.marker = marker
            self.plotting_name = plotting_name or patterns[0]
            self.decision_makers = []
            for subject in df.subj.unique():
                for pattern in patterns:
                    if fnmatch.fnmatch(subject, pattern) and subject not in self.decision_makers:
                        self.decision_makers.append(subject)

    decision_makers_mod.DecisionMaker = DecisionMaker
    decision_makers_mod.is_human_decision_maker = is_human_decision_maker
    sys.modules["neuroai.plotting.decision_makers"] = decision_makers_mod


_install_simple_eval_stubs()
simple_eval = _load_module("neuroai.simple_eval", ROOT / "neuroai" / "simple_eval.py")


class SimpleEvalPlotContextTests(unittest.TestCase):
    def test_model_context_only_keeps_target_model(self):
        dm_fun = simple_eval._make_simple_decision_maker(
            "resnet50_DVD_B",
            plotting_name="ours",
            plot_context="model",
        )

        decision_makers = dm_fun(
            _FakeDataFrame(["resnet50_DVD_B", "subject-01", "adults", "resnet50_baseline"])
        )

        self.assertEqual([dm.plotting_name for dm in decision_makers], ["ours"])

    def test_humans_context_includes_human_groups_without_other_models(self):
        dm_fun = simple_eval._make_simple_decision_maker(
            "resnet50_DVD_B",
            plot_context="humans",
        )

        decision_makers = dm_fun(
            _FakeDataFrame(
                ["resnet50_DVD_B", "subject-01", "subject-02", "adults", "10_12_years_old", "resnet50_baseline"]
            )
        )

        self.assertEqual(
            [dm.plotting_name for dm in decision_makers],
            ["resnet50_DVD_B", "humans(10-12 y/o)", "humans(adults)", "humans"],
        )

    def test_all_context_includes_other_models_too(self):
        dm_fun = simple_eval._make_simple_decision_maker(
            "resnet50_DVD_B",
            plot_context="all",
        )

        decision_makers = dm_fun(
            _FakeDataFrame(
                ["resnet50_DVD_B", "subject-01", "adults", "4_6_years_old", "resnet50_baseline"]
            )
        )

        self.assertEqual(
            [dm.plotting_name for dm in decision_makers],
            ["resnet50_DVD_B", "humans(4-6 y/o)", "humans(adults)", "humans", "resnet50_baseline"],
        )


if __name__ == "__main__":
    unittest.main()
