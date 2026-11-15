"""Motion-only external stoppage model. Never reads a reference file at inference."""
import numpy as np

from possession.pipeline import local_mean, velocity

WINDOWS = (.4, 2., 6.)


def motion_features(data, pitch, fps):
    xy = data[:, :, :2]*np.asarray(pitch)
    ball, players = xy[:, -1], xy[:, :-1]
    observed = np.isfinite(ball).all(axis=1)
    ball_speed = np.linalg.norm(velocity(ball, fps), axis=1)
    player_speed = np.linalg.norm(velocity(players, fps), axis=2)
    # Same fixed physical jump scale as the existing control extractor.
    reliable = observed & (~np.isfinite(ball_speed) | (ball_speed <= 45.))
    player_speed[player_speed > 12.] = np.nan
    valid = np.isfinite(player_speed)
    count = valid.sum(axis=1)
    mean_speed = np.divide(np.nansum(player_speed, axis=1), count,
                           out=np.full(len(data), np.nan), where=count > 0)
    quiet_fraction = np.divide(((player_speed < .8) & valid).sum(axis=1), count,
                               out=np.full(len(data), np.nan), where=count > 0)
    distances = np.linalg.norm(players-ball[:, None], axis=2)
    nearest = np.min(np.where(np.isfinite(distances), distances, np.inf), axis=1)
    nearest[~np.isfinite(nearest) | ~reliable] = np.nan
    boundary = np.maximum.reduce([-ball[:, 0], ball[:, 0]-pitch[0],
                                  -ball[:, 1], ball[:, 1]-pitch[1]])
    boundary[~reliable] = np.nan
    ball_speed[~reliable] = np.nan
    raw = dict(ball_speed=np.log1p(np.clip(ball_speed, 0, 45)),
               outside=np.log1p(np.clip(boundary, 0, 10)),
               nearest_player=np.log1p(np.clip(nearest, 0, 30)),
               player_speed=np.log1p(mean_speed), quiet_fraction=quiet_fraction,
               ball_observed=observed.astype(float), ball_reliable=reliable.astype(float),
               player_observed_fraction=np.isfinite(players).all(axis=2).mean(axis=1))
    columns, names = [], []
    for name, values in raw.items():
        for seconds in WINDOWS:
            average = local_mean(values, round(seconds*fps/2))
            columns.extend([average, (~np.isfinite(average)).astype(float)])
            names.extend([f'{name}_mean_{seconds}s', f'{name}_missing_{seconds}s'])
    return np.column_stack(columns), names


def degrade(data, fps, seed):
    """Synthetic faults independent of all references; never supplies extra positions."""
    rng = np.random.default_rng(seed)
    result = data.copy()
    n = len(data)
    missing = np.zeros(n, dtype=bool)
    # Random blocks until at least the prescribed target; overlap is permitted.
    while missing.mean() < .43:
        start = rng.integers(0, n)
        length = round(rng.choice([1, 3, 8])*fps)
        missing[start:min(n, start+length)] = True
    result[missing, -1, :2] = np.nan
    player_missing = rng.random(result[:, :-1, 0].shape) < .15
    result[:, :-1, :2][player_missing] = np.nan
    jump = (rng.random(n) < .01) & ~missing
    # Input coordinates are normalised: caller supplies pitch for offset below.
    return result, jump, rng.normal(0, 25, (int(jump.sum()), 2))


def degraded_data(data, pitch, fps, seed):
    result, jump, offset_m = degrade(data, fps, seed)
    result[jump, -1, :2] += offset_m/np.asarray(pitch)
    return result


def prepare(x, model):
    return (np.where(np.isfinite(x), x, model['imputation'])-model['mean'])/model['scale']


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -np.asarray(x)))


def fit(x, y, ridge):
    """Penalised Newton logistic fit; imputation/scaling fitted on training only."""
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Training requires explicit live and dead examples')
    imputation = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                           for col in x.T])
    clean = np.where(np.isfinite(x), x, imputation)
    mean, scale = clean.mean(axis=0), clean.std(axis=0)
    scale[scale < 1e-8] = 1.
    design = np.column_stack([np.ones(len(x)), (clean-mean)/scale])
    coefficient = np.zeros(design.shape[1])
    coefficient[0] = np.log(y.mean()/(1-y.mean()))
    regularisation = np.full(len(coefficient), ridge)
    regularisation[0] = 0.

    def objective(w):
        z = design@w
        return np.mean(np.logaddexp(0, z)-y*z)+.5*np.dot(regularisation*w, w)

    for _ in range(50):
        probability = sigmoid(design@coefficient)
        gradient = design.T@(probability-y)/len(y)+regularisation*coefficient
        hessian = (design.T*(probability*(1-probability)))@design/len(y)
        hessian += np.diag(regularisation+1e-8)
        step = np.linalg.solve(hessian, gradient)
        before, amount = objective(coefficient), 1.
        while amount > 1e-6 and objective(coefficient-amount*step) > before:
            amount /= 2
        coefficient -= amount*step
        if np.max(np.abs(amount*step)) < 1e-6:
            break
    return dict(imputation=imputation, mean=mean, scale=scale,
                intercept=coefficient[0], coefficient=coefficient[1:], ridge=ridge)


def logits(x, model):
    if not np.isfinite(model['coefficient']).all():
        raise ValueError('Invalid model coefficients')
    return model['intercept']+prepare(x, model)@model['coefficient']


def decode_binary(log_odds, penalty, fps):
    """Binary Viterbi; evidence scaled to protocol's 5 Hz regardless of input fps."""
    if not len(log_odds) or not np.isfinite(log_odds).all() or fps <= 0:
        raise ValueError('Expected nonempty finite logits and positive fps')
    score = np.array([0., log_odds[0]*5/fps])
    back = np.empty((len(log_odds), 2), dtype=np.int8)
    for i in range(1, len(log_odds)):
        transition = np.array([[score[0], score[1]-penalty],
                               [score[0]-penalty, score[1]]])
        back[i] = transition.argmax(axis=1)
        score = transition.max(axis=1)+[0., log_odds[i]*5/fps]
        score -= score.max()
    states = np.empty(len(log_odds), dtype=np.int8)
    states[-1] = score.argmax()
    for i in range(len(states)-1, 0, -1):
        states[i-1] = back[i, states[i]]
    return states


def predict(data, pitch, fps, model):
    x, names = motion_features(data, pitch, fps)
    if names != list(model['feature_names']):
        raise ValueError('Model feature schema mismatch')
    log_odds = logits(x, model)
    return decode_binary(log_odds, float(model['switch_penalty']), fps), log_odds


def metrics(reference, predicted, fps):
    known = reference >= 0
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (reference[known], predicted[known]), 1)
    f1 = np.divide(2*np.diag(cm), cm.sum(axis=0)+cm.sum(axis=1),
                   out=np.zeros(2), where=(cm.sum(axis=0)+cm.sum(axis=1)) > 0)
    result = dict(known_frames=int(known.sum()), confusion_alive_dead=cm.tolist(),
                  accuracy=float(np.trace(cm)/max(1, cm.sum())),
                  macro_f1=float(f1.mean()), dead_f1=float(f1[1]))
    timings = {}
    for state, name in ((0, 'resume'), (1, 'stop')):
        truth = np.flatnonzero(known[1:] & known[:-1] & (reference[1:] == state) &
                               (reference[1:] != reference[:-1]))+1
        estimated = np.flatnonzero((predicted[1:] == state) & (predicted[1:] != predicted[:-1]))+1
        # One-to-one maximum-cardinality matching with a fixed 2s tolerance.
        i, j, errors = 0, 0, []
        while i < len(truth) and j < len(estimated):
            delta = (estimated[j]-truth[i])/fps
            if delta < -2:
                j += 1
            elif delta > 2:
                i += 1
            else:
                errors.append(abs(delta))
                i += 1
                j += 1
        timings[name] = dict(reference_count=len(truth), predicted_count=len(estimated),
                             matched_within_2s=len(errors),
                             matched_mean_absolute_seconds=float(np.mean(errors)) if errors else None)
    result['transitions'] = timings
    return result
