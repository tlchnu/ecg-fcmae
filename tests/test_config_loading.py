"""YAML configuration validation tests."""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecg_project.utils.config import parse_args_with_config


class ConfigLoadingTests(unittest.TestCase):
    def test_cli_overrides_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("training:\n  epochs: 100\n", encoding="utf-8")
            parser = argparse.ArgumentParser()
            parser.add_argument("--config")
            parser.add_argument("--epochs", type=int, default=20)
            with patch.object(
                sys,
                "argv",
                ["test", "--config", str(config_path), "--epochs", "3"],
            ):
                args = parse_args_with_config(parser)
            self.assertEqual(args.epochs, 3)


if __name__ == "__main__":
    unittest.main()
