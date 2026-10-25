import unittest
import numpy as np
import pandas as pd
from dino_branch_train import UID, LABELS
from ensemble_reference import blend_complete


class EnsembleTests(unittest.TestCase):
    def frame(self, reverse=False):
        values=np.arange(4, dtype=float)
        if reverse: values=values[::-1]
        out=pd.DataFrame({c:values for c in LABELS})
        out.insert(0,UID,['a','b','c','d'])
        return out

    def test_alignment_and_class_weights(self):
        a=self.frame(); b=self.frame(True)
        result=blend_complete(a,a,b.iloc[::-1],b)
        pos=np.array([.25,.5,.75,1]);neg=pos[::-1]
        np.testing.assert_allclose(result['ACL'],.5*pos+.5*neg)
        np.testing.assert_allclose(result['Synovitis'],.8*pos+.2*neg)
        np.testing.assert_allclose(result['Lateral OA'],.2*pos+.8*neg)

    def test_reject_incomplete_branch(self):
        a=self.frame()
        with self.assertRaises(AssertionError):blend_complete(a,a,a,a.iloc[:2])
