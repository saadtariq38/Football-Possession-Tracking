"""Deterministic motion/tagging prototype. Does not load labels or video."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, asset_paths, source_frames

LABELS = np.array(['H', 'A', 'D'])


def load_tracking(data_dir=None):
    paths = asset_paths(data_dir=data_dir)
    meta = ET.parse(paths['atd_meta']).getroot()
    teams = list(meta.iter('Team'))
    team_ids = [t.get('id') for t in teams]
    if len(teams) != 2:
        raise ValueError('Expected two teams')
    slots = np.array([team_ids.index(p.get('teamId')) for p in meta.iter('Player')])
    n = source_frames()
    data = np.empty((n, len(slots)+1, 3), dtype=np.float64)
    count = 0
    with open(paths['atd_raw']) as handle:
        for index, line in enumerate(handle):
            frame, players, ball = line.strip().split(':')
            if index >= n or int(frame) != index+1:
                raise ValueError('Raw tracking frame IDs/count do not match video')
            values = np.fromstring(players.replace(';', ',')+','+ball, sep=',')
            if values.size != (len(slots)+1)*3:
                raise ValueError(f'Invalid slot count at raw frame {frame}')
            data[index] = values.reshape(-1, 3)
            count += 1
    if count != n:
        raise ValueError(f'Expected {n} frames, got {count}')
    return data, slots, [t.findtext('Name') for t in teams], paths


def local_mean(values, width):
    """Centred finite-observation average. No zero padding at recording edges."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    sums = np.r_[0., np.cumsum(np.where(valid, values, 0.))]
    counts = np.r_[0, np.cumsum(valid)]
    indices = np.arange(len(values))
    lo, hi = np.maximum(0, indices-width), np.minimum(len(values), indices+width+1)
    denominator = counts[hi]-counts[lo]
    return np.divide(sums[hi]-sums[lo], denominator,
                     out=np.full(len(values), np.nan), where=denominator>0)


def velocity(xy, fps):
    """Adjacent observed differences only; never differentiates across missing frames."""
    out = np.full_like(xy, np.nan)
    out[1:] = np.diff(xy, axis=0)*fps
    return out


def features(data, slots, pitch, fps, config):
    xy = data[:, :, :2]*np.array(pitch)
    ball, players = xy[:, -1], xy[:, :-1]
    observed = np.isfinite(ball).all(axis=1)
    ball_v = velocity(ball, fps)
    player_v = velocity(players, fps)
    ball_speed = np.linalg.norm(ball_v, axis=1)
    reliability = np.where(np.isfinite(ball_speed),
                           np.exp(-(ball_speed/config['ball_jump_scale_mps'])**2), 1.)
    reliability *= observed
    distances = np.linalg.norm(players-ball[:, None, :], axis=2)
    relative_speed = np.linalg.norm(player_v-ball_v[:, None, :], axis=2)
    # Co-motion strengthens a nearby candidate but is never required for contact.
    motion = np.where(np.isfinite(relative_speed),
                      np.exp(-relative_speed/config['motion_difference_scale_mps']), 0.)
    proximity = np.exp(-0.5*(distances/config['control_radius_m'])**2)
    scores = np.nan_to_num(proximity, nan=0.)*(0.5+0.5*motion)*reliability[:, None]
    width = max(0, round(config['control_support_seconds']*fps/2))
    control = np.column_stack([local_mean(scores[:, slots==t].max(axis=1), width)
                               for t in (0, 1)])
    nearest = np.column_stack([np.min(np.where(np.isfinite(distances[:, slots==t]),
                                              distances[:, slots==t], np.inf), axis=1)
                               for t in (0, 1)])
    # Signed margin from the closest pitch boundary: positive means outside.
    margin = np.maximum.reduce([-ball[:, 0], ball[:, 0]-pitch[0],
                                -ball[:, 1], ball[:, 1]-pitch[1]])
    outside = np.where(observed, np.clip(margin/config['boundary_softness_m'], 0, 1), 0.)
    outside *= reliability
    speeds = np.linalg.norm(player_v, axis=2)
    speeds = np.where(np.isfinite(speeds), speeds, 0.)
    valid_speed = np.isfinite(player_v).all(axis=2)
    collective = np.divide(speeds.sum(axis=1), valid_speed.sum(axis=1),
                           out=np.full(len(data), np.nan), where=valid_speed.sum(axis=1)>0)
    quiet = np.exp(-collective/config['quiet_players_scale_mps'])
    stationary = np.exp(-ball_speed/config['stationary_ball_scale_mps'])
    return dict(ball_observed=observed, ball_xy_m=ball, ball_velocity_mps=ball_v,
                ball_reliability=reliability, nearest_team_distance_m=nearest,
                control=control, outside=outside,
                stationary_quiet=np.nan_to_num(quiet*stationary, nan=0.))


def tag_evidence(path, team_names, n, fps, config):
    """Interpret only explicit phase semantics. Restart windows are not dead labels."""
    votes = np.zeros((n, 2))
    restart = np.zeros(n)
    attacking = {'BUILD UP', 'PROGRESSION', 'CREATING CHANCES', 'LONG BALL',
                 'ATTACKING TRANSITION', 'BALL IN FINAL THIRD', 'BALL IN THE BOX'}
    defending = {'DEFENDING IN MIDDLE THIRD', 'DEFENDING IN DEFENSIVE THIRD',
                 'DEFENDING IN ATTACKING THIRD', 'DEFENSIVE TRANSITION'}
    # Tags and metadata use different spellings for Athletic in this bundle.
    def normalise(name):
        return (name or '').casefold().replace('athletic', 'atletic').strip()
    names = [normalise(t) for t in team_names]
    context = max(1, round(config['restart_context_seconds']*fps))
    for event in ET.parse(path).iter('instance'):
        start = max(0, min(n, round(float(event.findtext('start'))*fps)))
        end = max(start, min(n, round(float(event.findtext('end'))*fps)))
        code = (event.findtext('code') or '').upper().strip()
        tags = {l.findtext('group'): l.findtext('text') for l in event.findall('label')}
        team = normalise(tags.get('Team'))
        if team in names and code in attacking | defending:
            side = names.index(team)
            if code in defending:
                side = 1-side
            votes[start:end, side] += 1
        if code == 'SET PIECES':
            # Soft context before and during restart; requires independent stoppage evidence.
            lo = max(0, start-context)
            restart[lo:start] = np.maximum(restart[lo:start], np.linspace(0, 1, start-lo))
            restart[start:end] = 1
    # Multiple overlapping tags should not multiply certainty.
    return np.clip(votes[:, 0]-votes[:, 1], -1, 1), restart


def continuity(control, fps, config):
    """Carry earlier control forward; bridge matching evidence on both sides of gaps."""
    n = len(control)
    winner = np.argmax(control, axis=1)
    strength = np.abs(control[:, 0]-control[:, 1])
    # A candidate requires one side to be nearer than one radius, without ambiguity.
    anchors = (control.max(axis=1) >= np.exp(-0.5)*0.5) & (strength > 0)
    ids = np.arange(n)
    left = np.maximum.accumulate(np.where(anchors, ids, -1))
    right = np.minimum.accumulate(np.where(anchors, ids, n)[::-1])[::-1]
    safe_left, safe_right = np.maximum(left, 0), np.minimum(right, n-1)
    sign = np.where(winner==0, 1., -1.)
    forward = np.where(left>=0, sign[safe_left]*strength[safe_left]*
                       np.exp(-(ids-safe_left)/(fps*config['continuity_decay_seconds'])), 0.)
    matching = (left>=0)&(right<n)&(winner[safe_left]==winner[safe_right])
    bridge = (sign[safe_left]*np.minimum(strength[safe_left], strength[safe_right])*
              np.exp(-(safe_right-safe_left)/(fps*config['bridge_decay_seconds'])))
    return np.where(matching, np.where(np.abs(bridge)>np.abs(forward), bridge, forward), forward)


def emissions(f, tag_margin, restart, fps, config):
    carry = continuity(f['control'], fps, config)
    team_margin = config['control_weight']*(f['control'][:, 0]-f['control'][:, 1])
    team_margin += config['continuity_weight']*carry + config['tag_weight']*tag_margin
    stoppage = np.maximum(f['outside'], f['stationary_quiet'])
    dead = config['outside_weight']*f['outside']
    dead += config['restart_stoppage_weight']*restart*stoppage
    dead -= config['dead_base_penalty'] + config['control_weight']*f['control'].max(axis=1)
    return np.column_stack([team_margin, -team_margin, dead]), carry


def logsumexp(values, axis):
    m = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.exp(values-m).sum(axis=axis, keepdims=True)), axis=axis)


def decode(evidence, penalty, marginals=True):
    """Viterbi sequence plus forward/backward model marginals; no fitting."""
    n, k = evidence.shape
    if not n or k != 3 or not np.isfinite(evidence).all():
        raise ValueError('Expected finite, nonempty N x 3 evidence')
    transition = np.full((k, k), -penalty)
    np.fill_diagonal(transition, 0.)
    # Normalised initial and transition log probabilities.
    transition -= logsumexp(transition, axis=1)[:, None]
    back = np.zeros((n, k), dtype=np.int8)
    score = evidence[0]-np.log(k)
    alpha = np.empty_like(evidence) if marginals else None
    if marginals:
        alpha[0] = score-logsumexp(score, axis=0)
    for i in range(1, n):
        options = score[:, None]+transition
        back[i] = options.argmax(axis=0)
        score = options.max(axis=0)+evidence[i]
        score -= score.max()
        if marginals:
            alpha[i] = evidence[i]+logsumexp(alpha[i-1, :, None]+transition, axis=0)
            alpha[i] -= logsumexp(alpha[i], axis=0)
    states = np.empty(n, dtype=np.int8)
    states[-1] = score.argmax()
    for i in range(n-1, 0, -1):
        states[i-1] = back[i, states[i]]
    if not marginals:
        return states, None
    beta = np.zeros(k)
    posterior = np.empty_like(evidence)
    for i in range(n-1, -1, -1):
        prob = alpha[i]+beta
        posterior[i] = np.exp(prob-logsumexp(prob, axis=0))
        if i:
            beta = logsumexp(transition+evidence[i][None, :]+beta[None, :], axis=1)
            beta -= logsumexp(beta, axis=0)
    return states, posterior


def run(config_path, output, data_dir=None, stoppage_model=None):
    config = json.loads(Path(config_path).read_text())
    expected = json.loads((Path(__file__).parent/'config.json').read_text())
    if set(config) != set(expected) or any(not isinstance(v, (float, int)) or
            not np.isfinite(v) or v <= 0 for v in config.values()):
        raise ValueError('Configuration must contain the documented positive finite parameters')
    print('Loading tactical tracking...', flush=True)
    data, slots, teams, paths = load_tracking(data_dir)
    fps = GAMES[GAME]['fps']
    print('Extracting motion, proximity and tagging evidence...', flush=True)
    f = features(data, slots, GAMES[GAME]['pitch'], fps, config)
    tag_margin, restart = tag_evidence(paths['smart_tagging'], teams, len(data), fps, config)
    evidence, carry = emissions(f, tag_margin, restart, fps, config)
    extra = {}
    if stoppage_model:
        from possession.stoppage import predict
        model = json.loads(Path(stoppage_model).read_text())
        frozen_control = json.loads(Path(__file__).with_name('config_external_control_v1.json').read_text())
        if config != frozen_control:
            raise ValueError('Stoppage integration requires the frozen external control configuration')
        dead, dead_logits = predict(data, GAMES[GAME]['pitch'], fps, model)
        extra.update(dead_state=dead, dead_log_odds=dead_logits,
                     original_emission=evidence.copy())
        # Independent live/dead detector; retain the frozen H/A evidence and decoder.
        evidence[:, 2] = -1e6
    print('Resolving sequence with past and future evidence...', flush=True)
    states, posterior = decode(evidence, config['switch_penalty'])
    if stoppage_model:
        extra['team_state'] = states.copy()
        states[dead == 1] = 2
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        handle.write('frame,label\n')
        for i, label in enumerate(LABELS[states]):
            handle.write(f'{i},{label}\n')
    intermediate = Path(data_dir or paths['atd_raw']).resolve()
    if not data_dir:
        intermediate = intermediate.parent.parent
    run_name = 'possession_v1' if output.stem == 'tactical_prototype' else output.stem
    intermediate = intermediate/'processed'/run_name
    intermediate.mkdir(parents=True, exist_ok=True)
    extra['team_model_marginals' if stoppage_model else 'model_marginals'] = posterior
    np.savez_compressed(intermediate/'evidence.npz', **f, **extra, tag_margin=tag_margin,
                        restart_context=restart, continuity=carry, emission=evidence,
                        state=states)
    hashes = {}
    for key in ('atd_meta', 'atd_raw', 'smart_tagging'):
        digest = hashlib.sha256()
        with open(paths[key], 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
        hashes[key] = digest.hexdigest()
    calibrated_path = Path(__file__).parent/'config_external_control_v1.json'
    calibrated = calibrated_path.exists() and config == json.loads(calibrated_path.read_text())
    report = dict(status=('Externally calibrated control candidate; dead-ball and tagging settings uncalibrated'
                         if calibrated else 'Uncalibrated prototype; accuracy not established'), config=config,
                  config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                  calibration_document='possession/CALIBRATION.md' if calibrated else None,
                  source_sha256=hashes, submission_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  frames=len(data), ball_observed=int(f['ball_observed'].sum()),
                  label_counts={l: int((LABELS[states]==l).sum()) for l in LABELS},
                  parameters=('Control radius, support, continuity decay and switch penalty selected on external development matches; other parameters fixed'
                              if calibrated else 'Fixed engineering assumptions; no fitting or tuning performed'),
                  marginals='Internal model support, not calibrated correctness probabilities')
    if stoppage_model:
        report.update(status='External control and motion-only stoppage candidate; challenge accuracy unverified',
                      stoppage_model_sha256=hashlib.sha256(Path(stoppage_model).read_bytes()).hexdigest(),
                      stoppage_protocol_sha256=model['protocol_sha256'],
                      stoppage_document='possession/STOPPAGE_CALIBRATION.md',
                      parameters='Control and motion stoppage calibrated on separate external datasets; tagging settings unchanged',
                      integration='Binary D overrides frozen H/A sequence; H/A is decoded across dead intervals',
                      marginals='Team-only internal marginals; dead_log_odds are raw logistic scores, not calibrated correctness probabilities')
    (intermediate/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Wrote {output}\nEvidence: {intermediate}\n{report["label_counts"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).parent/'config.json'))
    parser.add_argument('--out', default=str(ROOT/'evaluation/results/tactical_prototype.csv'))
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--stoppage-model', default=None)
    args = parser.parse_args()
    run(args.config, args.out, args.data_dir, args.stoppage_model)
