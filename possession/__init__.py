"""Our possession pipeline: tracking in, one frame,label CSV out.

    atd.py        the tactical tracking as numpy arrays, cached under data/cache/
    tagging.py    Smart Tagging as per-frame hints
    features.py   the per-frame measurements table every later step reads
    series.py     helpers for per-frame arrays: runs, rolling windows
    deadball.py   the stoppages: when the ball is out of play
    tracker.py    who has the ball between stoppages
    run.py        all of the above, written out as the submission CSV
"""
