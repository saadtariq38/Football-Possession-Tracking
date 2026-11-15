"""Synthetic checks only: no match labels or video."""
import itertools
import json
from pathlib import Path
import unittest

import numpy as np
from possession.pipeline import continuity, decode, features, logsumexp, tag_evidence

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'possession/config.json').read_text())


class PipelineTests(unittest.TestCase):
    def test_sequence_matches_exhaustive_search_and_marginals(self):
        evidence = np.array([[2., 0., -1.], [0., .2, 0.], [0., 2., -1.]])
        penalty = .8
        states, posterior = decode(evidence, penalty)
        paths = np.array(list(itertools.product(range(3), repeat=3)))
        weights = np.array([sum(evidence[i, s] for i, s in enumerate(p))-
                            penalty*np.count_nonzero(p[1:] != p[:-1]) for p in paths])
        expected = paths[weights.argmax()]
        np.testing.assert_array_equal(states, expected)
        weights = np.exp(weights-logsumexp(weights, axis=0))
        brute = np.array([[weights[paths[:, i]==s].sum() for s in range(3)] for i in range(3)])
        np.testing.assert_allclose(posterior, brute, atol=1e-12)

    def test_missing_ball_does_not_create_dead_ball_evidence(self):
        data = np.full((20, 3, 3), np.nan)
        f = features(data, np.array([0, 1]), (105.3, 68.), 25, CONFIG)
        self.assertFalse(f['ball_observed'].any())
        self.assertFalse(f['outside'].any())
        self.assertFalse(f['stationary_quiet'].any())
        self.assertFalse(f['control'].any())

    def test_short_same_team_gap_has_continuity(self):
        control = np.zeros((12, 2))
        control[0, 0] = .9
        control[-1, 0] = .9
        carry = continuity(control, 25, CONFIG)
        self.assertTrue(np.all(carry>0))
        # A second-team anchor must not be bridged as continuous home control.
        control[-1] = [0, .9]
        carry = continuity(control, 25, CONFIG)
        self.assertLess(carry[-1], 0)

    def test_motion_and_boundary_units(self):
        data = np.zeros((10, 3, 3))
        data[:, -1, 0] = np.arange(10)/105.3
        data[:, -1, 1] = .5
        f = features(data, np.array([0, 1]), (105.3, 68.), 25, CONFIG)
        np.testing.assert_allclose(f['ball_velocity_mps'][1:, 0], 25.)
        self.assertFalse(f['outside'].any())
        data[-1, -1, 0] = np.nan
        f = features(data, np.array([0, 1]), (105.3, 68.), 25, CONFIG)
        self.assertTrue(np.isnan(f['ball_velocity_mps'][-1]).any())

    def test_defensive_phase_votes_for_opposite_team(self):
        import tempfile
        xml = '''<root><instance><start>0</start><end>1</end>
        <code>DEFENDING IN MIDDLE THIRD</code>
        <label><group>Team</group><text>Away</text></label></instance></root>'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'tags.xml'
            path.write_text(xml)
            margin, restart = tag_evidence(path, ['Home', 'Away'], 50, 25, CONFIG)
        np.testing.assert_array_equal(margin[:25], 1.)
        self.assertFalse(restart.any())


if __name__ == '__main__':
    unittest.main()
