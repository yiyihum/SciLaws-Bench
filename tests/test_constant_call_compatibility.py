"""Regression checks for prompt-template and explicit-keyword submissions."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import eval_formula as ef
import evaluate_parallel as ep


def clustered_data():
    header = ["group_id", "x", "y"]
    rows = [["1", "1", "5"], ["1", "2", "7"], ["1", "3", "9"]]
    return dict(fit_header=header, test_header=header,
                fit_by_cluster={1: rows}, test_by_cluster={1: rows}, cluster_ids=[1])


class ConstantCompatibilityTests(unittest.TestCase):
    def test_global_style_runs_fit_and_prediction(self):
        law = {"slope": 2.0}
        calls = []
        def fit(X_fit, y_fit):
            return {"offset": float(np.mean(y_fit - law["slope"] * X_fit[:, 0]))}
        def predict(X, offset):
            calls.append(len(X))
            return law["slope"] * X[:, 0] + offset
        mod = SimpleNamespace(USED_INPUTS=["x"], LAW_CONSTANTS=law,
                              LOCAL_FITTABLE={"offset": {}}, fit=fit, predict=predict)
        result = ef.run_formula(mod, clustered_data(), "y")
        self.assertEqual(result["n_clusters_failed"], 0)
        self.assertEqual(result["per_cluster"][1]["metrics"]["rmse"], 0)
        self.assertEqual(calls, [3])

    def test_explicit_style_still_receives_shared_and_local_values(self):
        seen = []
        def fit(X_fit, y_fit, *, slope):
            seen.append(slope)
            return {"offset": float(np.mean(y_fit - slope * X_fit[:, 0]))}
        def predict(X, offset, *, slope):
            seen.append(slope)
            return slope * X[:, 0] + offset
        mod = SimpleNamespace(USED_INPUTS=["x"], LAW_CONSTANTS={"slope": 2.0},
                              LOCAL_FITTABLE={"offset": {}}, fit=fit, predict=predict)
        result = ef.run_formula(mod, clustered_data(), "y")
        self.assertEqual(result["per_cluster"][1]["metrics"]["rmse"], 0)
        self.assertEqual(seen, [2.0] * 2)

    def test_explicit_kwargs_are_preferred_even_if_global_form_would_bind(self):
        def fn(X, **kwargs):
            return kwargs
        self.assertEqual(ef.call_with_law_constants(fn, None, law_constants={"c": 2},
                         local_params={"a": 3}), {"c": 2, "a": 3})

    def test_internal_typeerror_is_not_retried_even_if_message_looks_like_binding(self):
        calls = []
        def fn(X, **kwargs):
            calls.append(kwargs)
            raise TypeError("inner() got an unexpected keyword argument 'c'")
        with self.assertRaisesRegex(TypeError, "inner"):
            ef.call_with_law_constants(fn, None, law_constants={"c": 2})
        self.assertEqual(calls, [{"c": 2}])

    def test_internal_fit_failure_remains_an_execution_error(self):
        calls = []
        def fit(X_fit, y_fit, **kwargs):
            calls.append(kwargs)
            raise TypeError("bug in fit body")
        mod = SimpleNamespace(USED_INPUTS=["x"], LAW_CONSTANTS={"slope": 2.0},
                              LOCAL_FITTABLE={"offset": {}}, fit=fit,
                              predict=lambda X, **kwargs: X[:, 0])
        result = ef.run_formula(mod, clustered_data(), "y")
        self.assertEqual(result["n_clusters_failed"], 1)
        self.assertIn("bug in fit body", result["per_cluster"][1]["error"])
        self.assertEqual(len(calls), 1)

    def test_local_parameters_are_never_silently_dropped(self):
        calls = []
        def predict(X):
            calls.append(1)
        with self.assertRaises(TypeError):
            ef.call_with_law_constants(predict, None, law_constants={"c": 2},
                                       local_params={"offset": 3})
        self.assertEqual(calls, [])

    def test_missing_required_parameters_are_not_repaired(self):
        def predict(X, required):
            self.fail("must not execute")
        with self.assertRaises(TypeError):
            ef.call_with_law_constants(predict, None, law_constants={"c": 2})

    def test_shared_local_name_collision_is_rejected(self):
        def predict(X, **kwargs):
            self.fail("must not execute")
        with self.assertRaisesRegex(TypeError, "overlap"):
            ef.call_with_law_constants(predict, None, law_constants={"c": 2},
                                       local_params={"c": 3})

    def test_uninspectable_function_keeps_original_explicit_call(self):
        def fn(X, **kwargs):
            return kwargs
        with patch.object(ef.inspect, "signature", side_effect=ValueError("unknown signature")):
            self.assertEqual(ef.call_with_law_constants(fn, None, law_constants={"c": 2}), {"c": 2})

    def test_empty_law_keeps_prompt_template_working(self):
        def fit(X, y):
            return {"a": 1}
        self.assertEqual(ef.call_with_law_constants(fit, None, None, law_constants={}), {"a": 1})

    def test_flat_global_style(self):
        flat = dict(test_header=["x", "y"], test_rows=[["1", "2"], ["2", "4"]])
        mod = SimpleNamespace(USED_INPUTS=["x"], LAW_CONSTANTS={"c": 2},
                              predict=lambda X: 2 * X[:, 0])
        self.assertEqual(ef.run_formula_flat(mod, flat, "y")["metrics"]["rmse"], 0)

    def test_parallel_contract_gate_accepts_both_styles(self):
        for explicit in [False, True]:
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as temporary:
                stage = Path(temporary)
                (stage / "metadata.yaml").write_text("type: typeII\n")
                fit_args = ", slope" if explicit else ""
                predict_args = ", slope" if explicit else ""
                value = "slope" if explicit else "LAW_CONSTANTS['slope']"
                (stage / "submission.py").write_text(
                    "USED_INPUTS = ['x']\nLAW_CONSTANTS = {'slope': 2.0}\n"
                    "OTHER_CONSTANTS = {}\nLOCAL_FITTABLE = {'offset': {}}\n"
                    f"def fit(X_fit, y_fit{fit_args}):\n    return {{'offset': 3.0}}\n"
                    f"def predict(X, offset{predict_args}):\n    return {value} * X[:, 0] + offset\n")
                with patch.object(ep, "_sample_matrix", return_value=np.arange(8.0).reshape(-1, 1)), \
                     patch.object(ep, "_sample_y", return_value=np.arange(8.0)):
                    self.assertEqual(ep._contract_check(stage, str(explicit)), (True, ""))


if __name__ == "__main__":
    unittest.main()
