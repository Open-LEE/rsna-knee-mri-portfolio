import unittest
import numpy as np
import pandas as pd
from dino_group_cv import assign_folds
from dino_branch_train import UID,LABELS


class FoldTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame([{UID:str(i),'group':i//2,'split':'train' if i<30 else 'gold_reference',
                              **{c:float(i%2) for c in LABELS}} for i in range(34)])
    def test_groups_disjoint_and_every_case_predicted_once(self):
        f=assign_folds(self.table());self.assertEqual(len(f),30)
        covered=[]
        for k in range(3):
            tr=f[f.fold!=k];va=f[f.fold==k]
            self.assertFalse(set(tr.group)&set(va.group));covered.extend(va[UID])
        self.assertEqual(len(set(covered)),30);self.assertEqual(len(covered),30)
    def test_fold_assignment_independent_of_labels(self):
        df=self.table();a=assign_folds(df)
        df[LABELS]=1-df[LABELS];b=assign_folds(df)
        pd.testing.assert_series_equal(a.fold,b.fold)
        pd.testing.assert_frame_equal(a,assign_folds(self.table()))


if __name__=='__main__':unittest.main()
