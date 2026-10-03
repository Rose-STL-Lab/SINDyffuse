from __future__ import annotations
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import ast

ROOT = Path(__file__).resolve().parent.parent

class OpenSimAdUnitsTest(unittest.TestCase):
    def test_nonfinite_nlp_fails_before_solver_and_finite_nlp_can_solve(self):
        import casadi as ca
        if not hasattr(ca, 'Opti'):
            self.skipTest('CasADi stubbed in suite; validated separately with real CasADi')
        tree = ast.parse((ROOT / 'nimble/opensimad/vendor/opencap_ad/utilsOpenSimAD.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_with_bounds')
        import os
        ns = {'np': np, 'ca': ca, 'os': os}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<solver-preflight>', 'exec'), ns)
        opti = ca.Opti(); x = opti.variable(); opti.set_initial(x, 0)
        opti.minimize(x*x); opti.subject_to(1/x == 1)
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite NLP initial evaluation'):
            ns['solve_with_bounds'](opti, 3, False)
        opti = ca.Opti(); x = opti.variable(); opti.set_initial(x, 0.5)
        opti.minimize((x-1)**2); opti.subject_to(x >= 0)
        result, stats = ns['solve_with_bounds'](opti, 3, False)
        self.assertTrue(stats['success'])
        self.assertAlmostEqual(float(result[0,0]), 1, places=3)

    def test_storage_metadata_matches_ik_units(self):
        spec = importlib.util.spec_from_file_location('storage_utils_test', ROOT / 'nimble/opensimad/vendor/opencap_ad/utils.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'opensim': types.ModuleType('opensim')}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'motion.mot'
            module.numpy_to_storage(['time', 'knee'], np.array([[0, 90.]]), path, datatype='IK')
            self.assertIn('inDegrees=yes', path.read_text())
            self.assertIn('90.00000000', path.read_text())
            module.numpy_to_storage(['time', 'force'], np.array([[0, 90.]]), path)
            self.assertIn('inDegrees=no', path.read_text())

    def test_spline_conversion_reverses_coefficients_and_reports_error(self):
        spec = importlib.util.spec_from_file_location('model_prep_test', ROOT / 'nimble/opensimad/model_prep.py')
        module = importlib.util.module_from_spec(spec)
        osim = MagicMock()
        x = np.linspace(-1, 2, 10); y = 2 + 3*x + 4*x**2
        spline = MagicMock()
        spline.getX.return_value.getSize.return_value = len(x)
        spline.getY.return_value.getSize.return_value = len(y)
        spline.getX.return_value.get.side_effect = lambda i: x[i]
        spline.getY.return_value.get.side_effect = lambda i: y[i]
        osim.SimmSpline.safeDownCast.return_value = spline
        osim.Vector.side_effect = lambda values: np.asarray(values)
        coefficients = []
        def polynomial(values):
            coefficients.append(values.copy())
            result = MagicMock()
            result.calcValue.side_effect = lambda argument: np.polyval(values, argument[0])
            return result
        osim.PolynomialFunction.side_effect = polynomial
        model = MagicMock()
        joints = model.get_JointSet.return_value
        joints.getSize.return_value = 1
        joint = joints.get.return_value
        joint.getConcreteClassName.return_value = 'CustomJoint'
        joint.getName.return_value = 'test_joint'
        transform = osim.CustomJoint.safeDownCast.return_value.get_SpatialTransform.return_value
        for name in ('rotation1','rotation2','rotation3','translation1','translation2','translation3'):
            getattr(transform, 'get_' + name).return_value.get_function.return_value.getConcreteClassName.return_value = 'Constant'
        transform.get_rotation1.return_value.get_function.return_value.getConcreteClassName.return_value = 'SimmSpline'
        with patch.dict(sys.modules, {'opensim': osim}):
            spec.loader.exec_module(module)
            report = module._replace_simm_splines_in_spatial_transforms(model)
        np.testing.assert_allclose(np.polyval(coefficients[0], x), y, atol=1e-12)
        self.assertLess(report[0]['knot_max_error'], 1e-12)
