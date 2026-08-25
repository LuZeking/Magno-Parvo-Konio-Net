import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return

    module = types.ModuleType(name)
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


_ensure_package("neuroai", ROOT / "neuroai")
_ensure_package("neuroai.helper", ROOT / "neuroai" / "helper")

c = _load_module("neuroai.constants", ROOT / "neuroai" / "constants.py")
ph = _load_module(
    "neuroai.helper.plotting_helper",
    ROOT / "neuroai" / "helper" / "plotting_helper.py",
)


class PlottingHelperPathResolutionTests(unittest.TestCase):
    def test_get_experiment_path_falls_back_to_repo_root_raw_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "repo"
            dataset_dir = repo_root / "raw-data" / "cue-conflict"
            dataset_dir.mkdir(parents=True)

            old_cwd = os.getcwd()
            os.chdir(tmp_path)
            try:
                with patch.object(c, "PROJ_DIR", str(repo_root)), patch.object(
                    c,
                    "RAW_DATA_DIR",
                    str(repo_root / "results" / "shape_bias" / "raw-data"),
                ):
                    experiment_path = ph.get_experiment_path("cue-conflict")
            finally:
                os.chdir(old_cwd)

            self.assertEqual(os.path.realpath(experiment_path), os.path.realpath(dataset_dir))

    def test_find_experiment_path_falls_back_to_current_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_root = tmp_path / "run"
            dataset_dir = run_root / "raw-data" / "cue-conflict"
            dataset_dir.mkdir(parents=True)

            old_cwd = os.getcwd()
            os.chdir(run_root)
            try:
                with patch.object(c, "PROJ_DIR", str(tmp_path / "installed-package")), patch.object(
                    c,
                    "RAW_DATA_DIR",
                    str(tmp_path / "installed-package" / "results" / "shape_bias" / "raw-data"),
                ):
                    experiment_path = ph.find_experiment_path("cue-conflict")
            finally:
                os.chdir(old_cwd)

            self.assertEqual(os.path.realpath(experiment_path), os.path.realpath(dataset_dir))

    def test_crop_pdfs_in_directory_skips_when_pdfcrop_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "figure.pdf").write_bytes(b"%PDF-1.4\n")

            with patch.object(ph.shutil, "which", return_value=None), patch.object(
                ph.warnings, "warn"
            ) as warn_mock:
                ph.crop_pdfs_in_directory(str(tmp_path))

            warn_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
