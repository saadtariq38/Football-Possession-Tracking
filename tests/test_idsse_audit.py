import tempfile
from pathlib import Path
import unittest
from scripts.audit_idsse_sample import audit


class RawStatusTests(unittest.TestCase):
    def test_unknown_status_and_frame_gaps_are_not_invented_dead(self):
        xml='''<root><FrameSet GameSection="firstHalf" TeamId="BALL">
        <Frame N="1" BallStatus="1"/><Frame N="2" BallStatus="0"/>
        <Frame N="3"/><Frame N="5" BallStatus="unexpected"/>
        </FrameSet></root>'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.xml';p.write_text(xml);r=audit(p)
        period=r['periods']['firstHalf']
        self.assertEqual(period['state_counts'],dict(alive=1,dead=1,unknown=2))
        self.assertEqual(period['missing_frame_ids'],1)
