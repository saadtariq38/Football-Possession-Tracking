import unittest
import numpy as np
from possession.skillcorner import encode_frame, seconds


class AdapterTests(unittest.TestCase):
    def test_observation_masks_unknown_labels_and_altitude(self):
        meta = dict(pitch_length=100, pitch_width=60)
        row = dict(period=1, possession=dict(group=None, player_id=None),
            player_data=[dict(player_id=7, x=0, y=-15, is_detected=False)],
            ball_data=dict(x=25, y=15, z=20, is_detected=True))
        data, flags, present, team, player = encode_frame(row, meta, {7:0})
        self.assertTrue(np.isnan(data[0]).all())
        self.assertEqual(flags.tolist(), [0, 1])
        self.assertEqual(present.tolist(), [True, True])
        np.testing.assert_allclose(data[-1, :2], [.75, .75])
        self.assertTrue(np.isnan(data[-1, 2]))
        self.assertEqual((team, player), (-1, -1))

    def test_reference_changes_cannot_change_inputs(self):
        meta=dict(pitch_length=100, pitch_width=60)
        row=dict(period=1, player_data=[], ball_data=dict(x=0,y=0,is_detected=True),
                 possession=dict(group='home team',player_id=7))
        first=encode_frame(row,meta,{7:0})
        row['possession']=dict(group='away team',player_id=None)
        second=encode_frame(row,meta,{7:0})
        for a,b in zip(first[:3],second[:3]):np.testing.assert_equal(a,b)
        self.assertEqual((first[3],second[3]),(0,1))

    def test_period_break_is_not_dead_label(self):
        row=dict(period=None,player_data=[],ball_data=dict(x=0,y=0,is_detected=True),
                 possession=dict(group='home team',player_id=7))
        data,_,_,team,player=encode_frame(row,dict(pitch_length=100,pitch_width=60),{7:0})
        self.assertTrue(np.isnan(data).all())
        self.assertEqual((team,player),(-1,-1))
        self.assertEqual(seconds('00:45:00.10'),2700.1)
