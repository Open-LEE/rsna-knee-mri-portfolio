import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import dino_branch_train as D


class DinoTests(unittest.TestCase):
    def test_group_isolation_and_official_precedence(self):
        rows=[]; weak=[]; hashes={}
        for i in range(100):
            uid=str(i);row={D.UID:uid,'Report':f'report {i}'}
            row.update({c:float(i%2) if i<4 else np.nan for c in D.LABELS})
            rows.append(row);weak.append({D.UID:uid,**{c+'__verdict':'YES' if i%2 else 'NO' for c in D.LABELS}});hashes[uid]=str(i)
        rows[4]['Report']=rows[0]['Report']
        hashes['5']=hashes['4']  # transitive image/report link to gold
        rows[7]['Report']=rows[6]['Report']
        result=D.make_split(pd.DataFrame(rows),pd.DataFrame(weak),hashes)
        self.assertEqual(result.loc[4,'split'],'excluded');self.assertEqual(result.loc[5,'split'],'excluded')
        self.assertEqual(result.loc[6,'split'],result.loc[7,'split'])
        self.assertEqual(result.loc[0,'ACL'],0.) # official unaffected by public table
        for _,g in result[result.split.ne('excluded')].groupby('group'):
            self.assertEqual(g.split.nunique(),1)
        pd.testing.assert_frame_equal(result,D.make_split(pd.DataFrame(rows),pd.DataFrame(weak),hashes))

    def test_unknown_labels_zero_gradient(self):
        z=torch.zeros(2,12,requires_grad=True); y=torch.full((2,12),float('nan'));y[:,0]=torch.tensor([0.,1.])
        D.loss_fn(z,y).backward()
        self.assertTrue(torch.equal(z.grad[:,1:],torch.zeros_like(z.grad[:,1:])))
        self.assertTrue((z.grad[:,0]!=0).all())

    def test_missing_plane_no_effect_and_cpu_training(self):
        torch.manual_seed(1); model=D.Head().eval()
        f=torch.randn(2,3,12,384);m=torch.tensor([[True,False,True],[True,True,False]])
        p=model(f,m);altered=f.clone();altered[~m]=100
        torch.testing.assert_close(p,model(altered,m))
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        y=torch.zeros(2,12);loss=D.loss_fn(p,y);loss.backward();opt.step()
        self.assertTrue(torch.isfinite(loss))
        with tempfile.TemporaryDirectory() as folder:
            for i in range(2):np.savez(Path(folder)/f'{i}.npz',f=f[i].numpy().astype('float16'),mask=m[i].numpy())
            df=pd.DataFrame([{D.UID:str(i),**{c:float(i) for c in D.LABELS}} for i in range(2)])
            loader=torch.utils.data.DataLoader(D.Features(df,Path(folder)),batch_size=2)
            score,pred=D.evaluate(model,loader,'cpu')
            self.assertEqual(pred.shape,(2,12));self.assertEqual(score['valid_classes'],12)


if __name__=='__main__':unittest.main()
