"""GunnersFC possession program: tracking data in, one frame,label CSV out.

    atd.py        reads the tracking file
    tagging.py    reads Smart Tagging
    features.py   step 1: measurements for every frame
    deadball.py   step 2: stoppages
    tracker.py    step 3: which team has the ball
    run.py        runs all steps and writes the CSV
    series.py     small helpers used by the steps
"""
