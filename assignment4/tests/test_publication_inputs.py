import json
import tempfile
import unittest
from pathlib import Path

from ml441_assignment4.records import canonical_hash
from publication_inputs import inspect_inputs


class PublicationInputsTests(unittest.TestCase):
    def test_missing_private_inputs_are_reported_without_creating_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = inspect_inputs(root)
            self.assertEqual(result["status"], "missing_private_inputs")
            self.assertIn("configs/primary_protocol.json", result["missing"])
            self.assertEqual(result["new_fits"], 0)
            self.assertEqual(list(root.iterdir()), [])

    def test_protocol_seal_and_external_path_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "configs").mkdir()
            protocol = {"splits": {"source": "local/splits.npz"}, "gate_evidence": []}
            path = root / "configs/primary_protocol.json"
            seal = root / "configs/primary_protocol.sha256"
            path.write_text(json.dumps(protocol))
            seal.write_text(canonical_hash(protocol))
            self.assertTrue(inspect_inputs(root)["protocol_seal_matches"])
            protocol["splits"]["source"] = "../outside.npz"
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "leaves the local"):
                inspect_inputs(root)
            protocol["splits"]["source"] = "local/changed.npz"
            path.write_text(json.dumps(protocol))
            self.assertEqual(inspect_inputs(root)["status"], "invalid_protocol_seal")


if __name__ == "__main__":
    unittest.main()
