import json
from pathlib import Path
import unittest
import contextlib
import io
import tempfile
from unittest.mock import patch

import numpy as np

from possession import restart, stoppage
from possession import restart_release
from possession.validate_restart import release_gate
from possession.run_restart import load_models, run as run_candidate

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'possession/config_external_control_v1.json').read_text())


class RestartTests(unittest.TestCase):
    def test_directional_windows_ignore_missing_and_clip_recording(self):
        values = np.array([1., np.nan, 3., 5.])
        np.testing.assert_equal(restart.sided_mean(values, 2, False), [1., 1., 3., 4.])
        np.testing.assert_equal(restart.sided_mean(values, 2, True), [1., 3., 4., 5.])

    def test_team_features_reverse_sign_when_teams_swap(self):
        rng = np.random.default_rng(3)
        data = np.full((100, 5, 3), np.nan)
        data[:, :, :2] = .5+np.cumsum(rng.normal(0, .0002, (100, 5, 2)), axis=0)
        data[10:30, -1] = np.nan
        slots = np.array([0, 0, 1, 1])
        first, names = restart.team_features(data, slots, [105., 68.], 25, CONFIG)
        second, swapped_names = restart.team_features(data, 1-slots, [105., 68.], 25, CONFIG)
        self.assertEqual(names, swapped_names)
        np.testing.assert_allclose(first, -second, atol=1e-10)

    def test_symmetric_training_has_no_fixed_home_advantage(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(300, 6))
        home = (x[:, 0]+.1*x[:, 1] > 0).astype(int)
        model = restart.fit_team(x, home, .01)
        np.testing.assert_allclose(stoppage.logits(x, model), -stoppage.logits(-x, model), atol=1e-10)
        self.assertGreater(((stoppage.logits(x, model) > 0) == home).mean(), .95)

    def test_predicted_dead_reset_removes_team_path_carry(self):
        odds = np.array([3., 3., -.1, -.1])
        dead = np.array([1, 1, 0, 0])
        continuous = restart.team_decode(odds, dead, 8, 5, 'continuous')
        reset = restart.team_decode(odds, dead, 8, 5, 'reset_at_predicted_dead')
        np.testing.assert_equal(continuous, [0, 0, 0, 0])
        np.testing.assert_equal(reset, [0, 0, 1, 1])

    def test_missing_observations_do_not_create_control_evidence(self):
        data = np.full((40, 5, 3), np.nan)
        tx, _ = restart.team_features(data, np.array([0, 0, 1, 1]), [105., 68.], 25, CONFIG)
        sx, names = restart.stop_features(data, [105., 68.], 25)
        np.testing.assert_equal(tx, np.zeros_like(tx))
        self.assertFalse(np.isinf(sx).any())
        self.assertEqual(names[-1], 'ball_missing_fraction_1s')
        np.testing.assert_equal(sx[:, -1], np.ones(len(data)))
        self.assertTrue(np.isnan(data).all())

    def test_external_missing_shift_affects_only_missing_fraction(self):
        x = np.array([[0., 0.], [0., 1.]])
        model = dict(imputation=np.zeros(2), mean=np.zeros(2), scale=np.ones(2),
                     coefficient=np.zeros(2), intercept=1.)
        head = dict(kind='learned', model=model, missing_shift=-.5)
        np.testing.assert_equal(restart.stop_logits(x, head, {}), [1., .5])

    def test_release_correction_is_bounded_and_never_creates_dead(self):
        dead = np.array([1]*10+[0]*10+[1]*10)
        head = dict(kind='learned', maximum_advance_seconds=1.,
                    maximum_dead_probability=.25, sustain_seconds=.4)
        corrected = restart_release.apply(dead, np.zeros(30, dtype=bool), np.zeros(30), 5, head)
        np.testing.assert_equal(corrected, [1]*5+[0]*15+[1]*10)
        self.assertTrue((corrected <= dead).all())
        observed = restart_release.apply(dead, np.ones(30, dtype=bool), np.zeros(30), 5, head)
        np.testing.assert_equal(observed, dead)
        region, until = restart_release.candidate_region(dead, np.zeros(30, dtype=bool), 5)
        self.assertEqual(int(region.sum()), 10)
        self.assertFalse(region[20:].any())

    def test_release_requires_sustained_confidence(self):
        dead = np.array([1]*10+[0]*10)
        probability = np.ones(20)
        probability[8] = 0.
        head = dict(kind='learned', maximum_advance_seconds=2.,
                    maximum_dead_probability=.25, sustain_seconds=.4)
        np.testing.assert_equal(restart_release.apply(dead, np.zeros(20, dtype=bool), probability, 5, head), dead)

    def test_fresh_gate_cannot_hide_regression_in_one_scenario(self):
        def scores(accuracy, f1, restart_accuracy, false_dead):
            return dict(whole=dict(accuracy=accuracy), dead_f1=f1,
                        offset_accuracy=restart_accuracy, missing_offset_false_dead_rate=false_dead)
        before = {s: scores(.9, .95, .7, .2) for s in ('original', 'degraded')}
        after = {s: scores(.91, .95, .75, .1) for s in ('original', 'degraded')}
        self.assertTrue(release_gate(before, after)['passed'])
        after['degraded']['whole']['accuracy'] = .89
        self.assertFalse(release_gate(before, after)['passed'])

    def test_frozen_candidate_runner_matches_inference_and_preserves_output(self):
        from games import GAME, GAMES
        bundle, refinement = load_models()
        rng = np.random.default_rng(5)
        n = 80
        data = np.full((n, 5, 3), np.nan)
        data[:, :, :2] = .5+np.cumsum(rng.normal(0, .0002, (n, 5, 2)), axis=0)
        slots = np.array([0, 0, 1, 1])
        tag = np.linspace(-.5, .5, n)
        expected = restart_release.predict(data, slots, GAMES[GAME]['pitch'], GAMES[GAME]['fps'],
                                            bundle, refinement, tag)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = {}
            for key in ('atd_meta', 'atd_raw', 'smart_tagging'):
                paths[key] = folder/f'{key}.txt'
                paths[key].write_text('synthetic fixture')
            with patch('possession.run_restart.load_tracking', return_value=(data, slots, ['H', 'A'], paths)), \
                 patch('possession.run_restart.tag_evidence', return_value=(tag, np.zeros(n))), \
                 contextlib.redirect_stdout(io.StringIO()):
                output = folder/'candidate.csv'
                destination = run_candidate(output, folder)
                actual = np.loadtxt(output, delimiter=',', skiprows=1, dtype=str)[:, 1]
                np.testing.assert_equal(actual, np.array(['H', 'A', 'D'])[expected['state']])
                with np.load(destination/'evidence.npz') as saved:
                    np.testing.assert_equal(saved['state'], expected['state'])
                with self.assertRaises(FileExistsError):
                    run_candidate(output, folder)
