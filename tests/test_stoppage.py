import itertools
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from possession.idsse import adapt
from possession.stoppage import (decode_binary, degraded_data, fit, logits,
                                  metrics, motion_features)


class StoppageTests(unittest.TestCase):
    def test_binary_decoder_matches_exhaustive_paths(self):
        evidence = np.array([2., -3., -1., 4., 2.])
        for fps in (5., 25.):
            for penalty in (0., 2., 8.):
                actual = decode_binary(evidence, penalty, fps)
                def score(path):
                    return np.dot(path, evidence)*5/fps-penalty*np.count_nonzero(np.diff(path))
                expected = max(score(path) for path in itertools.product((0, 1), repeat=len(evidence)))
                self.assertAlmostEqual(score(actual), expected)

    def test_logistic_fit_and_missing_inference(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(1000, 2))
        y = (x[:, 0] > .2).astype(int)
        x[::7, 1] = np.nan
        model = fit(x, y, .01)
        self.assertGreater(((logits(x, model) > 0) == y).mean(), .97)
        self.assertTrue(np.isfinite(logits(np.full((2, 2), np.nan), model)).all())

    def test_degradation_never_adds_observations_and_is_reproducible(self):
        data = np.ones((1000, 5, 3))*.5
        data[20:30, -1, :2] = np.nan
        a = degraded_data(data, [105., 68.], 25, 1)
        b = degraded_data(data, [105., 68.], 25, 1)
        np.testing.assert_equal(a, b)
        self.assertTrue(np.isnan(a[20:30, -1, :2]).all())
        x, names = motion_features(a, [105., 68.], 25)
        self.assertEqual(x.shape, (1000, len(names)))
        self.assertTrue(np.isfinite(x).any())

    def test_transition_matching_cannot_reuse_predictions(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0])
        pred = np.array([0, 0, 0, 1, 1, 1, 1])
        result = metrics(y, pred, 5)
        self.assertEqual(result['transitions']['stop']['matched_within_2s'], 1)

    def test_adapter_status_changes_do_not_change_inputs(self):
        meta = '''<root><General HomeTeamId="H" GuestTeamId="A"/>
        <Environment PitchX="100" PitchY="60"/>
        <Team TeamId="H"><Player PersonId="p"/></Team>
        <Team TeamId="A"><Player PersonId="q"/></Team></root>'''
        tracking = '''<root><FrameSet GameSection="firstHalf" TeamId="H" PersonId="p">
        <Frame N="1" T="2022-01-01T00:00:00+00:00" X="0" Y="0"/>
        <Frame N="3" T="2022-01-01T00:00:00.080+00:00" X="1" Y="0"/>
        </FrameSet><FrameSet GameSection="firstHalf" TeamId="BALL">
        <Frame N="1" T="2022-01-01T00:00:00+00:00" X="10" Y="15" Z="99" S="99" BallStatus="{status}" BallPossession="2"/>
        <Frame N="3" T="2022-01-01T00:00:00.080+00:00" X="11" Y="16"/>
        </FrameSet></root>'''
        inputs = []
        refs = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for status in ('0', '1'):
                source, output = root/status, root/f'out{status}'
                source.mkdir()
                m, t = source/'DFL_02_01_M.xml', source/'DFL_04_03_M.xml'
                m.write_text(meta)
                t.write_text(tracking.format(status=status))
                (source/'provenance.json').write_text(json.dumps(dict(files=[
                    dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (m, t)])))
                folder = adapt('M', source, output)
                inputs.append(dict(np.load(folder/'inputs.npz')))
                refs.append(np.load(folder/'reference.npz')['dead'])
        for key in inputs[0]:
            np.testing.assert_equal(inputs[0][key], inputs[1][key])
        np.testing.assert_allclose(inputs[0]['data'][0, -1], [.6, .75, np.nan])
        np.testing.assert_equal(inputs[0]['segments'], [[0, 1], [1, 2]])
        np.testing.assert_equal(refs[0], [1, -1])
        np.testing.assert_equal(refs[1], [0, -1])
