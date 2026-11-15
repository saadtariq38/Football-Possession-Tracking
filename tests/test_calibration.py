import unittest
import numpy as np
from possession.calibrate import metric
from possession.pipeline import decode


class CalibrationTests(unittest.TestCase):
    def test_unknown_excluded_dead_prediction_counts_as_miss(self):
        r=metric(np.array([0,1,-1]),np.array([0,2,1]))
        self.assertEqual(r['reference_frames'],2)
        self.assertEqual(r['accuracy_known_HA'],.5)
        self.assertEqual(r['macro_f1_known_HA'],.5)
        self.assertEqual(r['predicted_D_on_known_HA'],1)

    def test_fast_decoder_preserves_path(self):
        e=np.random.default_rng(42).normal(size=(100,3))
        a,_=decode(e,3.)
        b,p=decode(e,3.,marginals=False)
        np.testing.assert_array_equal(a,b)
        self.assertIsNone(p)
