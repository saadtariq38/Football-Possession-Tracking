"""Stream-audit explicit raw BallStatus; never manufacture D from missing attributes."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/external/idsse'


def audit(path):
    frame_ids=defaultdict(set)
    ball={}
    sets=Counter()
    raw_status=Counter()
    examples=[]
    stack=[]
    section=None
    is_ball=False
    for event,elem in ET.iterparse(path,events=('start','end')):
        if event=='start':
            stack.append(elem)
            if elem.tag=='FrameSet':
                section=elem.get('GameSection')
                is_ball=elem.get('TeamId')=='BALL'
                sets[f'{section}:ball' if is_ball else f'{section}:player']+=1
        else:
            if elem.tag=='Frame':
                frame=int(elem.attrib['N'])
                frame_ids[section].add(frame)
                if is_ball:
                    key=(section,frame)
                    if key in ball:raise ValueError('Duplicate ball frame')
                    status=elem.get('BallStatus')
                    raw_status[str(status)]+=1
                    label={'0':'dead','1':'alive'}.get(status,'unknown')
                    ball[key]=(label,elem.get('T'),elem.get('X') is not None and elem.get('Y') is not None)
                    if len(examples)<3:examples.append(dict(elem.attrib))
                if len(stack)>1:stack[-2].remove(elem)
                elem.clear()
            elif elem.tag=='FrameSet':
                if len(stack)>1:stack[-2].remove(elem)
                elem.clear()
                section=None
                is_ball=False
            stack.pop()
    periods={}
    for period,ids in frame_ids.items():
        ordered=sorted(ids)
        labels=Counter(ball.get((period,i),('unknown',None,False))[0] for i in ordered)
        transitions=Counter()
        delta=Counter()
        missing_xy=Counter()
        run_lengths=defaultdict(list)
        prev_id=None;prev_label=None;length=0
        for i in ordered:
            label,t,has_xy=ball.get((period,i),('unknown',None,False))
            if not has_xy:missing_xy[label]+=1
            if prev_id is not None and i==prev_id+1:
                if prev_label!=label:transitions[f'{prev_label}->{label}']+=1
                old=ball.get((period,prev_id),(None,None,False))[1]
                if old and t:
                    dt=(datetime.fromisoformat(t)-datetime.fromisoformat(old)).total_seconds()
                    delta[str(round(dt,6))]+=1
            if prev_id is not None and (i!=prev_id+1 or label!=prev_label):
                run_lengths[prev_label].append(length);length=0
            length+=1;prev_id=i;prev_label=label
        if length:run_lengths[prev_label].append(length)
        periods[period]=dict(frame_count=len(ids),min_frame=min(ids),max_frame=max(ids),
            missing_frame_ids=max(ids)-min(ids)+1-len(ids),state_counts=dict(labels),
            missing_xy_by_state=dict(missing_xy),transitions=dict(transitions),
            timestamp_steps_seconds=dict(delta),
            runs={k:dict(count=len(v),max_frames=max(v),mean_frames=sum(v)/len(v)) for k,v in run_lengths.items()})
    return dict(raw_ball_status_values=dict(raw_status),frame_sets=dict(sets),periods=periods,
                raw_ball_examples=examples,
                status_mapping='Raw explicit 1=alive, 0=dead per Kloppy Sportec parser; missing/other=unknown',
                purpose='Schema and coverage only; labels not supplied as algorithm inputs; no tuning')


def main():
    path=next(FOLDER.glob('*positions*J03WPY.xml'))
    result=audit(path)
    (FOLDER/'audit_J03WPY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
