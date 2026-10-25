import unittest

import numpy as np
import pandas as pd
from dino_candidate_blend import blend,UID
from dino_branch_train import LABELS


class CandidateTests(unittest.TestCase):
    def base(self):
        return pd.DataFrame({UID:['a','b','c'],**{c:[.2,.5,.8] for c in LABELS}})
    def test_fixed_weight_and_alignment(self):
        base=self.base();d=base.iloc[[2,0,1]].copy();d.loc[:,LABELS]=np.array([[.1]*12,[.9]*12,[.5]*12])
        out,r=blend(base,d)
        np.testing.assert_allclose(out['ACL'],.95*np.array([.2,.5,.8])+.05*np.array([1,2/3,1/3]))
        self.assertEqual(r['blended_rows'],3)
    def test_missing_invalid_and_singleton_preserve_baseline(self):
        base=self.base()
        for d in [base.iloc[:0],base.iloc[:1],base.assign(ACL=float('nan'))]:
            out,r=blend(base,d);pd.testing.assert_frame_equal(base,out);self.assertEqual(r['blended_rows'],0)
        d=base.iloc[:2].copy();out,r=blend(base,d)
        pd.testing.assert_series_equal(out.iloc[2],base.iloc[2])
        self.assertEqual(r['baseline_only_rows'],1)
    def test_unknown_or_duplicate_ids_rejected(self):
        base=self.base()
        for d in [pd.concat([base,base.iloc[:1]]),base.assign(StudyInstanceUID=['x','b','c'])]:
            with self.assertRaises(AssertionError):blend(base,d)
    def test_cv_partial_coverage_preserves_entire_baseline(self):
        from dino_cv_candidate_blend import blend as cv_blend
        base=self.base()
        out,receipt=cv_blend(base,base.iloc[:2])
        pd.testing.assert_frame_equal(out,base)
        self.assertEqual(receipt['blended_rows'],0)

if __name__=='__main__':unittest.main()
