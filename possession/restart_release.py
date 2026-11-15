"""Bounded, observation-only release correction around predicted restart boundaries."""
import numpy as np

from possession import restart, stoppage
from possession.joint_metrics import run_starts


def candidate_region(dead, observed, fps):
    region = np.zeros(len(dead), dtype=bool)
    until_resume = np.zeros(len(dead))
    for resume in np.flatnonzero((dead[:-1] == 1) & (dead[1:] == 0))+1:
        lo = int(resume)-1
        lower_bound = max(0, int(resume)-round(2*fps))
        while lo > lower_bound and dead[lo-1] == 1:
            lo -= 1
        ids = np.arange(lo, resume)
        region[ids] = ~observed[ids]
        until_resume[ids] = (resume-ids)/fps/2
    return region, until_resume


def feature_matrix(data, pitch, fps, base_dead, base_log_odds):
    sx, names = restart.stop_features(data, pitch, fps)
    selected_names = ['player_speed_past_0.4s', 'player_speed_future_0.4s',
        'player_speed_past_1.2s', 'player_speed_future_1.2s',
        'player_active_past_0.4s', 'player_active_future_0.4s',
        'ball_observed_future_1.2s']
    observed = np.isfinite(data[:, -1, :2]).all(axis=1)
    region, until = candidate_region(base_dead, observed, fps)
    x = np.column_stack([base_log_odds, sx[:, [names.index(n) for n in selected_names]], until])
    return x, ['base_dead_log_odds']+selected_names+['time_until_predicted_resume_over_2s'], region, observed


def apply(dead, observed, probability_dead, fps, head):
    result = dead.copy()
    if head['kind'] == 'none':
        return result
    width = max(1, round(head['sustain_seconds']*fps))
    for resume in np.flatnonzero((dead[:-1] == 1) & (dead[1:] == 0))+1:
        lo = int(resume)-1
        lower = max(0, int(resume)-round(head['maximum_advance_seconds']*fps))
        while lo > lower and dead[lo-1] == 1:
            lo -= 1
        eligible = ((dead[lo:resume] == 1) & ~observed[lo:resume] &
                    (probability_dead[lo:resume] < head['maximum_dead_probability']))
        starts = run_starts(eligible, width)
        if len(starts):
            result[lo+starts[0]:resume] = 0
    return result


def predict(data, slots, pitch, fps, bundle, refinement, tag_margin=None):
    base = restart.predict(data, slots, pitch, fps, bundle, tag_margin)
    x, names, region, observed = feature_matrix(data, pitch, fps, base['dead'], base['dead_log_odds'])
    if names != refinement['feature_names']:
        raise ValueError('Release feature schema mismatch')
    head = refinement['head']
    probability = (np.ones(len(data)) if head['kind'] == 'none' else
                   stoppage.sigmoid(stoppage.logits(x, head['model'])))
    dead = apply(base['dead'], observed, probability, fps, head)
    team = base['team'].copy()
    th = bundle['team_head']
    if th['kind'] == 'learned' and th['mode'] == 'reset_at_predicted_dead':
        team = restart.team_decode(base['home_log_odds'], dead, th['penalty'], fps, th['mode'])
    state = team.copy()
    state[dead == 1] = 2
    return dict(base, state=state, team=team, dead=dead, base_state=base['state'],
                base_dead=base['dead'], release_probability_dead=probability,
                release_candidate_region=region, release_changed=base['dead'] != dead)
