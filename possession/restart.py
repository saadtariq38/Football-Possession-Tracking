"""External restart candidate: directional motion and symmetric team evidence."""
import numpy as np

from possession import stoppage
from possession.pipeline import continuity, features, local_mean, velocity


def sided_mean(values, width, future):
    values = np.asarray(values, dtype=float)
    width = max(1, int(width))
    ids = np.arange(len(values))
    lo = ids if future else np.maximum(0, ids-width+1)
    hi = np.minimum(len(values), ids+width) if future else ids+1
    valid = np.isfinite(values)
    total = np.r_[0., np.cumsum(np.where(valid, values, 0.))]
    count = np.r_[0, np.cumsum(valid)]
    denominator = count[hi]-count[lo]
    return np.divide(total[hi]-total[lo], denominator,
                     out=np.full(len(values), np.nan), where=denominator > 0)


def stop_features(data, pitch, fps):
    base, names = stoppage.motion_features(data, pitch, fps)
    xy = data[:, :, :2]*np.asarray(pitch)
    observed = np.isfinite(xy[:, -1]).all(axis=1)
    ball_speed = np.linalg.norm(velocity(xy[:, -1], fps), axis=1)
    ball_speed[ball_speed > 45] = np.nan
    player_speed = np.linalg.norm(velocity(xy[:, :-1], fps), axis=2)
    player_speed[player_speed > 12] = np.nan
    valid = np.isfinite(player_speed)
    count = valid.sum(axis=1)
    mean = np.divide(np.nansum(player_speed, axis=1), count,
                     out=np.full(len(data), np.nan), where=count > 0)
    active = np.divide(((player_speed > 2) & valid).sum(axis=1), count,
                       out=np.full(len(data), np.nan), where=count > 0)
    columns = [base]
    raw = dict(player_speed=np.log1p(mean), player_active=active,
               ball_speed=np.log1p(ball_speed), ball_observed=observed.astype(float))
    for name, value in raw.items():
        for seconds in (.4, 1.2, 3.):
            for future, direction in ((False, 'past'), (True, 'future')):
                average = sided_mean(value, round(seconds*fps), future)
                columns.extend([average[:, None], (~np.isfinite(average))[:, None]])
                names.extend([f'{name}_{direction}_{seconds}s', f'{name}_{direction}_{seconds}s_missing'])
    missing = 1-local_mean(observed.astype(float), round(fps/2))
    columns.append(missing[:, None])
    names.append('ball_missing_fraction_1s')
    return np.column_stack(columns), names


def team_features(data, slots, pitch, fps, config):
    """Every column reverses sign when home/away slots are exchanged."""
    xy = data[:, :, :2]*np.asarray(pitch)
    ball, players = xy[:, -1], xy[:, :-1]
    ball_v, player_v = velocity(ball, fps), velocity(players, fps)
    ball_speed = np.linalg.norm(ball_v, axis=1)
    observed = np.isfinite(ball).all(axis=1)
    reliability = observed*np.where(np.isfinite(ball_speed), np.exp(-(ball_speed/45)**2), 1.)
    distances = np.linalg.norm(players-ball[:, None], axis=2)

    def difference(scores):
        scores = np.nan_to_num(scores, nan=0., posinf=0., neginf=0.)
        return scores[:, slots == 0].max(axis=1)-scores[:, slots == 1].max(axis=1)

    core = {}
    for radius in (1., 2., 4.):
        core[f'proximity_{radius}m'] = difference(np.exp(-.5*(distances/radius)**2))*reliability
    relative_speed = np.linalg.norm(player_v-ball_v[:, None], axis=2)
    co_motion = np.exp(-.5*(distances/2)**2)*np.exp(-relative_speed/8)
    core['co_motion'] = difference(co_motion)*reliability
    nearest = [np.min(np.where(np.isfinite(distances[:, slots == side]),
                    distances[:, slots == side], np.inf), axis=1) for side in (0, 1)]
    core['nearest_difference'] = (np.clip(nearest[1], 0, 30)-np.clip(nearest[0], 0, 30))/30*reliability
    # Constant-relative-velocity closest approach in the next two seconds.
    offset, relative_v = ball[:, None]-players, ball_v[:, None]-player_v
    time = np.clip(-np.sum(offset*relative_v, axis=2)/
                   (np.sum(relative_v**2, axis=2)+1e-6), 0, 2)
    miss = np.linalg.norm(offset+time[:, :, None]*relative_v, axis=2)
    core['approach'] = difference(np.exp(-.5*(miss/2)**2)*np.exp(-time))*reliability
    f = features(data, slots, pitch, fps, config)
    core['control'] = f['control'][:, 0]-f['control'][:, 1]
    core['past_anchor'] = continuity(f['control'], fps, config)
    core['future_anchor'] = continuity(f['control'][::-1], fps, config)[::-1]
    columns, names = [], []
    for name, value in core.items():
        for seconds in (.4, 1.2, 3.):
            for direction in ('centre', 'past', 'future'):
                averaged = (local_mean(value, round(seconds*fps/2)) if direction == 'centre' else
                            sided_mean(value, round(seconds*fps), direction == 'future'))
                columns.append(averaged)
                names.append(f'{name}_{direction}_{seconds}s')
    missing = 1-local_mean(observed.astype(float), round(.6*fps))
    fast = local_mean(np.nan_to_num(np.clip(ball_speed/20, 0, 1), nan=0.), round(.6*fps))
    for name in ('proximity_2.0m', 'past_anchor', 'future_anchor'):
        for context, values in (('missing', missing), ('fast', fast)):
            columns.append(core[name]*values)
            names.append(f'{name}_times_{context}')
    return np.nan_to_num(np.column_stack(columns), nan=0., posinf=0., neginf=0.), names


def fit_team(x, home, ridge):
    # Interleaved team-swapped examples remove an intercept or team identity advantage.
    paired = np.stack([x, -x], axis=1).reshape(-1, x.shape[1]).astype(float)
    target = np.stack([home, 1-home], axis=1).reshape(-1)
    model = stoppage.fit(paired, target, ridge)
    if abs(model['intercept']) > 1e-6 or np.max(np.abs(model['mean'])) > 1e-6:
        raise ValueError('Symmetric training lost team-swap invariance')
    # Remove floating point residuals only, not learned offsets.
    model['intercept'] = 0.
    model['mean'] = np.zeros_like(model['mean'])
    model['imputation'] = np.zeros_like(model['imputation'])
    return model


def team_decode(home_log_odds, dead, penalty, fps, mode):
    if mode not in ('continuous', 'reset_at_predicted_dead'):
        raise ValueError('Unknown team sequence mode')
    team = 1-stoppage.decode_binary(home_log_odds, penalty, fps)
    if mode == 'reset_at_predicted_dead':
        edges = np.r_[0, np.flatnonzero(dead[1:] != dead[:-1])+1, len(dead)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            if dead[lo] == 0:
                team[lo:hi] = 1-stoppage.decode_binary(home_log_odds[lo:hi], penalty, fps)
    return team


def stop_logits(x, head, v1_model):
    if head['kind'] == 'v1':
        return stoppage.logits(x[:, :len(v1_model['feature_names'])], v1_model)
    result = stoppage.logits(x, head['model'])
    return result+head['missing_shift']*x[:, -1]


def predict(data, slots, pitch, fps, bundle, tag_margin=None):
    sx, sn = stop_features(data, pitch, fps)
    tx, tn = team_features(data, slots, pitch, fps, bundle['control_config'])
    if sn != bundle['stop_feature_names'] or tn != bundle['team_feature_names']:
        raise ValueError('Restart model feature schema differs')
    sh, th = bundle['stop_head'], bundle['team_head']
    dead_odds = stop_logits(sx, sh, bundle['v1_stoppage_model'])
    dead = stoppage.decode_binary(dead_odds, sh['penalty'], fps)
    tag_margin = np.zeros(len(data)) if tag_margin is None else np.asarray(tag_margin)
    if th['kind'] == 'legacy':
        from possession.pipeline import emissions, decode
        f = features(data, slots, pitch, fps, bundle['control_config'])
        evidence, _ = emissions(f, tag_margin, np.zeros(len(data)), fps, bundle['control_config'])
        home_odds = evidence[:, 0]-evidence[:, 1]
        evidence[:, 2] = -1e6
        team, _ = decode(evidence, bundle['control_config']['switch_penalty'], marginals=False)
    else:
        home_odds = stoppage.logits(tx, th['model'])+1.2*tag_margin
        team = team_decode(home_odds, dead, th['penalty'], fps, th['mode'])
    state = team.copy()
    state[dead == 1] = 2
    return dict(state=state, team=team, dead=dead, home_log_odds=home_odds,
                dead_log_odds=dead_odds, ball_observed=np.isfinite(data[:, -1, :2]).all(axis=1))
