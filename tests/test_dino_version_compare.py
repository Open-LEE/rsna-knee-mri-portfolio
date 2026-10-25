import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from dino_version_compare import compare


class VersionTests(unittest.TestCase):
    def fixture(self,root,change=0.):
        for n in ['v025_a','v025_b','v029']:
            p=root/n;p.mkdir()
            prob=np.array([[.2,.3],[.7,.8]])
            if n=='v029':prob[0,0]+=change
            np.savez(p/'predictions.npz',prob=prob,features=np.zeros((2,3,12,4)),uids=np.array(['a','b']),labels=np.array(['ACL','MCL']))
            (p/'receipt.json').write_text(json.dumps({'checkpoint_sha256':'x','input_sha256':['a','b'],'torch':'same','gpu':'same'}))
    def test_equal_versions(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root)
            r=compare(root);self.assertEqual(r['cross_version_max_probability_difference'],0)
            self.assertFalse(r['difference_above_threshold'])
    def test_detect_difference(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root,.01)
            r=compare(root);self.assertTrue(r['difference_above_threshold'])
            self.assertAlmostEqual(r['per_class_max_probability_difference']['ACL'],.01)
    def test_mismatched_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root)
            p=root/'v029/receipt.json';r=json.loads(p.read_text());r['input_sha256']=['wrong'];p.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):compare(root)


if __name__=='__main__':unittest.main()
