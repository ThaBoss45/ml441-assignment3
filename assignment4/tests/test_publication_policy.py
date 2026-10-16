import ast
import io
import json
import tokenize
import unittest
from pathlib import Path


class PublicationPolicy(unittest.TestCase):
    def test_sources_have_no_comments_or_docstrings(self):
        root = Path(__file__).resolve().parents[1]
        paths = [*root.glob("src/ml441_assignment4/*.py"), *root.glob("src/*.py"), *root.glob("tests/*.py")]
        self.assertGreater(len(paths), 0)
        for path in paths:
            source = path.read_text()
            with self.subTest(path=str(path.relative_to(root))):
                tokens = tokenize.generate_tokens(io.StringIO(source).readline)
                self.assertFalse(any(token.type == tokenize.COMMENT for token in tokens))
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertIsNone(ast.get_docstring(node))

    def test_notebook_contains_only_clean_code_cells(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/experiment_log.ipynb").read_text())
        self.assertGreater(len(notebook["cells"]), 0)
        for cell in notebook["cells"]:
            self.assertEqual(cell["cell_type"], "code")
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            source = "".join(cell["source"])
            self.assertFalse(any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(source).readline)))
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertIsNone(ast.get_docstring(node))
        configuration = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("AUTOMATIC_COMMITS = False", configuration)
        self.assertIn("AUTOMATIC_PUSHES = False", configuration)
        self.assertIn("RUN_PILOT = False", configuration)
        self.assertIn("RUN_PRIMARY = False", configuration)

    def test_notebook_defaults_run_with_missing_private_inputs(self):
        import contextlib
        import os
        import tempfile
        from unittest.mock import patch

        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/experiment_log.ipynb").read_text())
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"ML441_A4_PROJECT_ROOT": folder}), contextlib.redirect_stdout(io.StringIO()):
            namespace = {"__name__": "__notebook_review__"}
            for cell in notebook["cells"]:
                exec(compile("".join(cell["source"]), "notebook_cell", "exec"), namespace)
            self.assertFalse(namespace["RUN_PILOT"])
            self.assertFalse(namespace["RUN_PRIMARY"])
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
