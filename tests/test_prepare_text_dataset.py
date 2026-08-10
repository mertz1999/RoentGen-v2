import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from roentgenv2.text_generation.prepare_dataset import build_source_text, main, report_key_from_stem


class PrepareTextDatasetTests(unittest.TestCase):
    def test_report_id_mapping_handles_shown_numeric_filenames(self):
        self.assertEqual(report_key_from_stem("1002"), "CXR1002")
        self.assertEqual(report_key_from_stem("CXR1002"), "CXR1002")
        self.assertEqual(report_key_from_stem("CXR1002_IM-1234-1001"), "CXR1002")
        self.assertEqual(report_key_from_stem("CXR1_1_IM-0001-3001"), "CXR1")

    def test_source_excludes_target_and_audit_metadata(self):
        source = build_source_text(
            {
                "indication": "Cough",
                "findings": "Clear lungs.",
                "impression": "No acute disease.",
                "mesh_labels": "normal",
                "num_images": 2,
                "generated_prompt": "must not appear",
                "cost_usd": 2.5,
            },
            ["indication", "findings", "impression", "mesh_labels", "num_images"],
        )
        self.assertIn("Indication: Cough", source)
        self.assertIn("Number of images: 2", source)
        self.assertNotIn("must not appear", source)
        self.assertNotIn("cost", source.lower())

    def test_external_test_folder_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = []
            for identifier in range(1, 5):
                rows.append(
                    {
                        "report_id": f"CXR{identifier}",
                        "indication": "cough",
                        "findings": "clear lungs",
                        "impression": "normal",
                        "mesh_labels": "normal",
                        "num_images": 1,
                        "generated_prompt": f"Prompt {identifier}.",
                    }
                )
            csv_file = root / "prompts.csv"
            pd.DataFrame(rows).to_csv(csv_file, index=False)
            database = root / "xray-database"
            for split, identifiers in (("train", (1, 2, 3)), ("test", (4,))):
                for identifier in identifiers:
                    (database / split / "images").mkdir(parents=True, exist_ok=True)
                    (database / split / "labels").mkdir(parents=True, exist_ok=True)
                    (database / split / "images" / f"{identifier}.png").touch()
                    (database / split / "labels" / f"{identifier}.txt").write_text(f"Prompt {identifier}.")
            output = root / "prepared"
            with patch.object(
                sys,
                "argv",
                [
                    "prepare_dataset.py",
                    "--csv-file",
                    str(csv_file),
                    "--xray-database",
                    str(database),
                    "--output-dir",
                    str(output),
                    "--validation-fraction",
                    "0.34",
                ],
            ):
                main()
            test_row = json.loads((output / "test.jsonl").read_text().strip())
            self.assertEqual(test_row["report_id"], "CXR4")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["examples"], {"train": 2, "validation": 1, "test": 1})


if __name__ == "__main__":
    unittest.main()
