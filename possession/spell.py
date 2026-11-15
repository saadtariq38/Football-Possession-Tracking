"""Spell-first temporal decoding, using only frozen external observation models."""
import numpy as np

from possession import restart, stoppage
from possession.pipeline import features, emissions, decode, local_mean


def binary_path(odds, penalty, fps):
    """Scalar implementation of the frozen binary Viterbi recurrence, for grid speed."""
    odds = np.asarray(odds)
    if not len(odds) or not np.isfinite(odds).all() or penalty < 0 or fps <= 0:
        raise ValueError('Finite nonempty scores and nonnegative cost required')
    score0, score1 = 0., float(odds[0])*5/fps
    back = np.empty((len(odds), 2), dtype=np.int8)
    for i in range(1, len(odds)):
        if score0 >= score1-penalty:
            next0, back[i, 0] = score0, 0
        else:
            next0, back[i, 0] = score1-penalty, 1
        if score0-penalty >= score1:
            next1, back[i, 1] = score0-penalty, 0
        else:
            next1, back[i, 1] = score1, 1
        next1 += float(odds[i])*5/fps
        normaliser = max(next0, next1)
        score0, score1 = next0-normaliser, next1-normaliser
    result = np.empty(len(odds), dtype=np.int8)
    result[-1] = int(score1 > score0)
    for i in range(len(result)-1, 0, -1):
        result[i-1] = back[i, result[i]]
    return result


def team_path(odds, dead, penalty, fps, mode):
    if mode not in ('continuous', 'reset_at_predicted_dead'):
        raise ValueError('Invalid sequence mode')
    team = 1-binary_path(odds, penalty, fps)
    if mode == 'reset_at_predicted_dead':
        edges = np.r_[0, np.flatnonzero(dead[1:] != dead[:-1])+1, len(dead)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            if dead[lo] == 0:
                team[lo:hi] = 1-binary_path(odds[lo:hi], penalty, fps)
    return team


def extract(data, slots, pitch, fps, bundle, tag_margin=None):
    config = bundle['control_config']
    tag = np.zeros(len(data)) if tag_margin is None else np.asarray(tag_margin)
    f = features(data, slots, pitch, fps, config)
    evidence, _ = emissions(f, tag, np.zeros(len(data)), fps, config)
    legacy_odds = evidence[:, 0]-evidence[:, 1]
    evidence[:, 2] = -1e6
    legacy_team, _ = decode(evidence, config['switch_penalty'], marginals=False)
    dead, dead_odds = stoppage.predict(data, pitch, fps, bundle['v1_stoppage_model'])
    tx, names = restart.team_features(data, slots, pitch, fps, config)
    if names != bundle['team_feature_names'] or bundle['team_head']['kind'] != 'learned':
        raise ValueError('Expected the frozen learned v2 team head')
    learned_odds = stoppage.logits(tx, bundle['team_head']['model'])+1.2*tag
    return dict(legacy_home_log_odds=legacy_odds, learned_home_log_odds=learned_odds,
                v1_team=legacy_team, dead=dead, dead_log_odds=dead_odds,
                ball_reliability=f['ball_reliability'],
                ball_observed=np.isfinite(data[:, -1, :2]).all(axis=1))


def apply(evidence, config, fps):
    if config['kind'] == 'v1':
        team = evidence['v1_team'].copy()
        odds = evidence['legacy_home_log_odds']
    elif config['kind'] == 'hybrid':
        quality = local_mean(evidence['ball_reliability'], round(.4*fps))
        fraction = config['learned_weight']*(quality+(1-quality)*config['missing_multiplier'])
        odds = fraction*evidence['learned_home_log_odds']+(1-fraction)*evidence['legacy_home_log_odds']
        if config['smoothing_seconds']:
            odds = local_mean(odds, round(config['smoothing_seconds']*fps/2))
        team = team_path(odds, evidence['dead'], config['switch_penalty'], fps, config['mode'])
    else:
        raise ValueError('Unknown spell configuration')
    state = team.copy()
    state[evidence['dead'] == 1] = 2
    return dict(state=state, team=team, dead=evidence['dead'], home_log_odds=odds)


def predict(data, slots, pitch, fps, bundle, config, tag_margin=None):
    evidence = extract(data, slots, pitch, fps, bundle, tag_margin)
    return dict(evidence, **apply(evidence, config, fps))
