"""Reference-side H/A/D metrics; these helpers are never called by inference."""
import numpy as np

LABELS = ('H', 'A', 'D')
STRATA = ('all', 'same_owner', 'changed_owner', 'unknown_owner')


def frame_metrics(reference, predicted, mask=None):
    if reference.shape != predicted.shape or not np.isin(predicted, [0, 1, 2]).all():
        raise ValueError('Aligned references and H/A/D predictions required')
    known = np.isin(reference, [0, 1, 2])
    selected = np.ones(len(reference), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    valid = known & selected
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (reference[valid], predicted[valid]), 1)
    denominator = cm.sum(axis=0)+cm.sum(axis=1)
    f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(3), where=denominator > 0)
    return dict(selected_frames=int(selected.sum()), known_frames=int(valid.sum()),
                excluded_unknown=int((selected & ~known).sum()),
                confusion_reference_rows_predicted_columns_HAD=cm.tolist(),
                accuracy=float(np.trace(cm)/cm.sum()) if cm.sum() else None,
                macro_f1_HAD=float(f1.mean()) if cm.sum() else None,
                per_class={label: dict(support=int(cm[i].sum()), f1=float(f1[i]))
                           for i, label in enumerate(LABELS)})


def merge_frame_metrics(parts):
    cm = np.sum([p['confusion_reference_rows_predicted_columns_HAD'] for p in parts], axis=0)
    denominator = cm.sum(axis=0)+cm.sum(axis=1)
    f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(3), where=denominator > 0)
    return dict(selected_frames=sum(p['selected_frames'] for p in parts),
                known_frames=int(cm.sum()), excluded_unknown=sum(p['excluded_unknown'] for p in parts),
                confusion_reference_rows_predicted_columns_HAD=cm.tolist(),
                accuracy=float(np.trace(cm)/cm.sum()) if cm.sum() else None,
                macro_f1_HAD=float(f1.mean()) if cm.sum() else None,
                per_class={label: dict(support=int(cm[i].sum()), f1=float(f1[i]))
                           for i, label in enumerate(LABELS)})


def transition_indices(dead):
    # Neither unknown statuses nor recording edges create events.
    restarts = np.flatnonzero((dead[:-1] == 1) & (dead[1:] == 0))+1
    stops = np.flatnonzero((dead[:-1] == 0) & (dead[1:] == 1))+1
    return restarts, stops


def window_mask(anchors, n, fps, before, after):
    mask = np.zeros(n, dtype=bool)
    for index in anchors:
        lo, hi = max(0, index-round(before*fps)), min(n, index+round(after*fps))
        mask[lo:hi] = True
    return mask


def restart_events(dead, reference, fps):
    restarts, _ = transition_indices(dead)
    events = []
    for index in restarts:
        dead_start = int(index)-1
        while dead_start > 0 and dead[dead_start-1] == 1:
            dead_start -= 1
        end = int(index)
        while end < len(dead) and dead[end] == 0:
            end += 1
        previous = reference[max(0, dead_start-round(fps)):dead_start]
        previous_dead = dead[max(0, dead_start-round(fps)):dead_start]
        previous = previous[(previous_dead == 0) & np.isin(previous, [0, 1])]
        current = reference[index:min(end, index+round(fps))]
        current = current[np.isin(current, [0, 1])]
        old_team = int(previous[-1]) if len(previous) else -1
        new_team = int(current[0]) if len(current) else -1
        stratum = ('unknown_owner' if min(old_team, new_team) < 0 else
                   'same_owner' if old_team == new_team else 'changed_owner')
        events.append(dict(index=int(index), preceding_dead_start=dead_start, live_end=end,
                           previous_team=old_team, first_restart_team=new_team, stratum=stratum))
    return events


def run_starts(mask, width):
    if len(mask) < width:
        return np.array([], dtype=int)
    cumulative = np.r_[0, np.cumsum(mask)]
    return np.flatnonzero(cumulative[width:]-cumulative[:-width] == width)


def restart_diagnostics(events, reference, predicted, fps, offsets, horizon=5., sustain=.4):
    result = {}
    width = max(1, round(sustain*fps))
    for group in STRATA:
        chosen = [e for e in events if group == 'all' or e['stratum'] == group]
        offset_metrics = {}
        for offset in offsets:
            eligible = correct = false_dead = wrong_previous = 0
            for event in chosen:
                index = event['index']+round(offset*fps)
                if index >= event['live_end'] or reference[index] not in (0, 1):
                    continue
                eligible += 1
                correct += int(predicted[index] == reference[index])
                false_dead += int(predicted[index] == 2)
                wrong_previous += int(event['stratum'] == 'changed_owner' and
                                      predicted[index] == event['previous_team'] and
                                      predicted[index] != reference[index])
            offset_metrics[str(offset)] = dict(eligible=eligible, correct=correct,
                accuracy=correct/eligible if eligible else None, false_dead=false_dead,
                wrong_team=eligible-correct-false_dead, wrong_previous_owner=wrong_previous)
        latencies, eligible_recovery = [], 0
        for event in chosen:
            lo = event['index']
            hi = min(event['live_end'], lo+round(horizon*fps))
            target = reference[lo:hi]
            known = np.isin(target, [0, 1])
            if not len(run_starts(known, width)):
                continue
            eligible_recovery += 1
            starts = run_starts(known & (predicted[lo:hi] == target), width)
            if len(starts):
                latencies.append(float(starts[0]/fps))
        result[group] = dict(events=len(chosen), offsets_seconds=offset_metrics,
            recovery=dict(eligible=eligible_recovery, excluded_no_known_sustained_reference=len(chosen)-eligible_recovery,
                recovered=len(latencies), censored=eligible_recovery-len(latencies),
                recovered_fraction=len(latencies)/eligible_recovery if eligible_recovery else None,
                recovered_latency_seconds=latencies))
    return result


def merge_restart_diagnostics(parts):
    result = {}
    for group in STRATA:
        selected = [p[group] for p in parts]
        offsets = {}
        for offset in selected[0]['offsets_seconds']:
            rows = [p['offsets_seconds'][offset] for p in selected]
            counts = {key: sum(r[key] for r in rows) for key in
                      ('eligible', 'correct', 'false_dead', 'wrong_team', 'wrong_previous_owner')}
            counts['accuracy'] = counts['correct']/counts['eligible'] if counts['eligible'] else None
            offsets[offset] = counts
        rows = [p['recovery'] for p in selected]
        recovery = {key: sum(r[key] for r in rows) for key in
                    ('eligible', 'excluded_no_known_sustained_reference', 'recovered', 'censored')}
        latencies = sum([r['recovered_latency_seconds'] for r in rows], [])
        recovery.update(recovered_fraction=recovery['recovered']/recovery['eligible'] if recovery['eligible'] else None,
                        recovered_mean_seconds=float(np.mean(latencies)) if latencies else None,
                        recovered_median_seconds=float(np.median(latencies)) if latencies else None,
                        recovered_latency_seconds=latencies)
        result[group] = dict(events=sum(p['events'] for p in selected), offsets_seconds=offsets, recovery=recovery)
    return result


def error_decomposition(reference, combined, team):
    alive = np.isin(reference, [0, 1])
    dead = reference == 2
    return dict(known_alive=int(alive.sum()), known_dead=int(dead.sum()),
                team_only_correct_alive=int((alive & (team == reference)).sum()),
                combined_correct_alive=int((alive & (combined == reference)).sum()),
                false_dead_on_alive=int((alive & (combined == 2)).sum()),
                wrong_team_on_alive=int((alive & (combined != 2) & (combined != reference)).sum()),
                missed_dead=int((dead & (combined != 2)).sum()),
                correct_team_overridden_dead=int((alive & (team == reference) & (combined == 2)).sum()))
