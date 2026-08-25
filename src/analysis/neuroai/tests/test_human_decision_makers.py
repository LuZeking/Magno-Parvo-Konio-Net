import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return

    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _ensure_matplotlib_stub() -> None:
    if "matplotlib" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.__path__ = []
        sys.modules["matplotlib"] = matplotlib

    if "matplotlib.lines" not in sys.modules:
        lines = types.ModuleType("matplotlib.lines")

        class Line2D:
            markers = {"o": None, "D": None, "v": None}

        lines.Line2D = Line2D
        sys.modules["matplotlib.lines"] = lines


def _load_module(name: str, path: Path):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ensure_matplotlib_stub()
_ensure_package("neuroai", ROOT / "neuroai")
_ensure_package("neuroai.plotting", ROOT / "neuroai" / "plotting")

_load_module("neuroai.constants", ROOT / "neuroai" / "constants.py")
_load_module("neuroai.plotting.colors", ROOT / "neuroai" / "plotting" / "colors.py")
dm = _load_module(
    "neuroai.plotting.decision_makers",
    ROOT / "neuroai" / "plotting" / "decision_makers.py",
)


class HumanDecisionMakerTests(unittest.TestCase):
    def test_is_human_decision_maker_recognizes_subjects_and_age_cohorts(self):
        self.assertTrue(dm.is_human_decision_maker("subject-01"))
        self.assertTrue(dm.is_human_decision_maker("adults"))
        self.assertTrue(dm.is_human_decision_maker("10_12_years_old"))
        self.assertFalse(dm.is_human_decision_maker("resnet50_DVD_B"))

    def test_get_human_and_model_decision_makers_treats_age_groups_as_humans(self):
        decision_makers = [
            SimpleNamespace(decision_makers=["subject-01", "adults"]),
            SimpleNamespace(decision_makers=["10_12_years_old", "resnet50_DVD_B"]),
        ]

        humans, models = dm.get_human_and_model_decision_makers(decision_makers)

        self.assertEqual(humans, ["subject-01", "adults", "10_12_years_old"])
        self.assertEqual(models, ["resnet50_DVD_B"])


if __name__ == "__main__":
    unittest.main()
