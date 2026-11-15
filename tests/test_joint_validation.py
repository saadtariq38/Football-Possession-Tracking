import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from games import GAME, GAMES
from possession.idsse_joint_reference import joint_state
from possession.joint_metrics import (frame_metrics, merge_frame_metrics, restart_diagnostics,
    restart_events, transition_indices, window_mask, error_decomposition)
from possession.validate_joint import infer_segment
from possession.pipeline import run

ROOT = Path(__file__).resolve().parents[1]


class JointValidationTests(unittest.TestCase):
    def test_unknown_reference_is_never_default_away_or_dead(self):
        for status, possession, expected in [
            ('0', None, 2), ('0', '1', 2), ('1', '1', 0), ('1', '2', 1),
            ('1', '0', -1), ('1', None, -1), (None, '1', -1), ('unexpected', '2', -1)]:
            frame = {}
            if status is not None:
                frame['BallStatus'] = status
            if possession is not None:
                frame['BallPossession'] = possession
            self.assertEqual(joint_state(frame), expected)

    def test_metrics_mask_unknown_and_merge_confusion_counts(self):
        reference = np.array([0, 1, 2, -1])
        predicted = np.array([0, 2, 1, 2])
        result = frame_metrics(reference, predicted)
        self.assertEqual(result['known_frames'], 3)
        self.assertEqual(result['excluded_unknown'], 1)
        self.assertAlmostEqual(result['accuracy'], 1/3)
        pieces = [frame_metrics(reference[a:b], predicted[a:b]) for a, b in ((0, 2), (2, 4))]
        self.assertEqual(merge_frame_metrics(pieces), result)
        errors = error_decomposition(reference, predicted, np.array([0, 1, 1, 0]))
        self.assertEqual(errors['false_dead_on_alive'], 1)
        self.assertEqual(errors['missed_dead'], 1)
        self.assertEqual(errors['correct_team_overridden_dead'], 1)

    def test_unknown_status_and_segment_edges_create_no_transition(self):
        dead = np.array([1, -1, 0, 1, 0])
        restarts, stops = transition_indices(dead)
        np.testing.assert_equal(restarts, [4])
        np.testing.assert_equal(stops, [3])
        # The same transition split across two source segments is not an event.
        self.assertEqual(len(transition_indices(np.array([1]))[0]), 0)
        self.assertEqual(len(transition_indices(np.array([0]))[0]), 0)

    def test_window_union_clips_and_deduplicates(self):
        mask = window_mask([1, 3], 6, 1, 2, 2)
        np.testing.assert_equal(mask, [True, True, True, True, True, False])

    def test_restart_offsets_do_not_cross_next_stoppage(self):
        reference = np.array([0, 0, 2, 2, 1, 1, 2])
        dead = (reference == 2).astype(int)
        events = restart_events(dead, reference, 2)
        self.assertEqual(events[0]['stratum'], 'changed_owner')
        result = restart_diagnostics(events, reference, np.array([0, 0, 2, 2, 0, 2, 2]),
                                     2, [0., .5, 1.])
        offsets = result['changed_owner']['offsets_seconds']
        self.assertEqual(offsets['0.0']['wrong_previous_owner'], 1)
        self.assertEqual(offsets['0.5']['false_dead'], 1)
        self.assertEqual(offsets['1.0']['eligible'], 0)

    def test_recovery_requires_sustained_correct_known_reference(self):
        reference = np.array([0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 2])
        dead = (reference == 2).astype(int)
        events = restart_events(dead, reference, 5)
        predicted = np.array([0, 0, 2, 2, 0, 0, 1, 1, 1, 1, 2])
        result = restart_diagnostics(events, reference, predicted, 5, [0.], 1.2, .4)
        self.assertEqual(result['all']['recovery']['recovered_latency_seconds'], [.4])
        censored = restart_diagnostics(events, reference, np.zeros(len(dead), dtype=int),
                                        5, [0.], 1.2, .4)
        self.assertEqual(censored['all']['recovery']['censored'], 1)
        unknown = reference.copy()
        unknown[4:10] = -1
        excluded = restart_diagnostics(events, unknown, predicted, 5, [0.], 1.2, .4)
        self.assertEqual(excluded['all']['recovery']['eligible'], 0)

    def test_external_inference_matches_production_pipeline(self):
        # Synthetic coordinates only, no match reference input; exercise the actual run path.
        n = 100
        rng = np.random.default_rng(12)
        data = np.full((n, 5, 3), np.nan)
        data[:, :, :2] = .5+np.cumsum(rng.normal(0, .0002, (n, 5, 2)), axis=0)
        slots = np.array([0, 0, 1, 1])
        config_path = ROOT/'possession/config_external_control_v1.json'
        model_path = ROOT/'possession/model_external_stoppage_v1.json'
        config, model = json.loads(config_path.read_text()), json.loads(model_path.read_text())
        predictions = infer_segment(data, slots, GAMES[GAME]['pitch'], GAMES[GAME]['fps'], config, model)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = {}
            for key in ('atd_meta', 'atd_raw', 'smart_tagging'):
                paths[key] = folder/f'{key}.txt'
                paths[key].write_text('synthetic source fixture')
            with patch('possession.pipeline.load_tracking', return_value=(data, slots, ['H', 'A'], paths)), \
                 patch('possession.pipeline.tag_evidence', return_value=(np.zeros(n), np.zeros(n))), \
                 contextlib.redirect_stdout(io.StringIO()):
                for name, selected_model in (('legacy_control', None), ('current_combined', model_path)):
                    output = folder/f'{name}.csv'
                    run(config_path, output, folder, selected_model)
                    actual = np.loadtxt(output, delimiter=',', skiprows=1, dtype=str)[:, 1]
                    np.testing.assert_equal(actual, np.array(['H', 'A', 'D'])[predictions[name]])
