import itertools
import unittest

import numpy as np

from evaluation import possession_metrics as pm
from possession import spell, stoppage
from possession.calibrate_spell import score
from possession.validate_spell import fresh_gate


class SpellTests(unittest.TestCase):
    def test_fast_decoder_matches_frozen_decoder(self):
        rng = np.random.default_rng(34)
        for fps, penalty in itertools.product((5., 25.), (0., 2., 8., 20.)):
            for evidence in (np.zeros(20), rng.normal(size=100), np.tile([2., -2.], 20)):
                np.testing.assert_equal(spell.binary_path(evidence, penalty, fps),
                                        stoppage.decode_binary(evidence, penalty, fps))

    def test_unreliable_ball_can_route_to_legacy_evidence(self):
        n = 30
        evidence = dict(legacy_home_log_odds=np.full(n, 2.), learned_home_log_odds=np.full(n, -4.),
                        ball_reliability=np.zeros(n), dead=np.zeros(n, dtype=int), v1_team=np.zeros(n, dtype=int))
        config = dict(kind='hybrid', learned_weight=1., missing_multiplier=0.,
                      smoothing_seconds=0., switch_penalty=0., mode='continuous')
        np.testing.assert_equal(spell.apply(evidence, config, 25)['team'], np.zeros(n))
        config['missing_multiplier'] = 1.
        np.testing.assert_equal(spell.apply(evidence, config, 25)['team'], np.ones(n))

    def test_switch_cost_prevents_short_spurious_possession(self):
        odds = np.r_[np.full(30, 2.), [-2., -2.], np.full(30, 2.)]
        raw = spell.team_path(odds, np.zeros(len(odds)), 0., 5, 'continuous')
        smooth = spell.team_path(odds, np.zeros(len(odds)), 8., 5, 'continuous')
        reference = np.zeros(len(odds), dtype=int)
        self.assertEqual(pm.spell_scores(reference, raw, [(0, len(odds))])['n_pred'], 3)
        self.assertEqual(pm.spell_scores(reference, smooth, [(0, len(odds))])['mean_f1'], 1.)

    def test_scoring_preserves_half_and_unknown_boundaries(self):
        pieces = [dict(match='M', scenario=scenario, y=np.array([0, 0, -1, 0]),
                       dead_reference=np.array([0, 0, -1, 0]), observed=np.ones(4, dtype=bool), fps=25.)
                  for scenario in ('original', 'degraded') for _ in range(2)]
        predicted = [np.zeros(4, dtype=int) for _ in pieces]
        scores = score(pieces, predicted)
        for result in scores.values():
            self.assertEqual(result['n_gt_spells'], 4)
            self.assertEqual(result['n_pred_spells'], 4)
            self.assertEqual(result['spell_f1'], 1.)

    def test_fresh_gate_checks_ranking_and_missing_restart_guards(self):
        baseline = {name: dict(spell_f1=.5, macro_f1=.9, missing_offset_accuracy=.4)
                    for name in ('M/original', 'M/degraded')}
        candidate = {name: dict(spell_f1=.55, macro_f1=.91, missing_offset_accuracy=.4)
                     for name in baseline}
        self.assertTrue(fresh_gate(baseline, candidate)['passed'])
        candidate['M/degraded']['missing_offset_accuracy'] = .37
        self.assertFalse(fresh_gate(baseline, candidate)['passed'])
        candidate['M/degraded']['missing_offset_accuracy'] = .4
        candidate['M/degraded']['spell_f1'] = .48
        self.assertFalse(fresh_gate(baseline, candidate)['passed'])

    def test_exact_v1_fallback_matches_frozen_inference(self):
        from possession.run_restart import load_models
        from possession.validate_joint import infer_segment
        bundle, _ = load_models()
        rng = np.random.default_rng(46)
        data = np.full((100, 5, 3), np.nan)
        data[:, :, :2] = .5+np.cumsum(rng.normal(0, .0002, (100, 5, 2)), axis=0)
        slots = np.array([0, 0, 1, 1])
        expected = infer_segment(data, slots, [105., 68.], 25, bundle['control_config'],
                                 bundle['v1_stoppage_model'])['current_combined']
        actual = spell.predict(data, slots, [105., 68.], 25, bundle, dict(kind='v1'))
        np.testing.assert_equal(actual['state'], expected)
