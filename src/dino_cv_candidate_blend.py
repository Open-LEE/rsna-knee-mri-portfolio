"""Fixed 5% rank blend; missing DINO predictions preserve baseline exactly."""
import numpy as np
import pandas as pd

UID='StudyInstanceUID'


def blend(base,dino,weight=.02):
    if not 0<=weight<=1:raise ValueError('weight')
    assert base[UID].notna().all() and base[UID].is_unique
    cols=[c for c in base if c!=UID]
    values=base[cols].to_numpy(float)
    assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
    assert dino[UID].notna().all() and dino[UID].is_unique
    assert set(dino[UID])<=set(base[UID]) and set(dino.columns)==set(base.columns)
    aligned=dino.set_index(UID).reindex(base[UID])[cols]
    v=aligned.to_numpy(float)
    good=np.isfinite(v).all(1)&((v>=0)&(v<=1)).all(1)
    result=base.copy()
    if len(base)>=2 and good.all():
        for j,c in enumerate(cols):
            # The existing baseline is already a weighted percentile-rank score.
            ranks=pd.Series(v[good,j]).rank(method='average',pct=True).to_numpy()
            result.loc[good,c]=(1-weight)*values[good,j]+weight*ranks
    return result,{'weight':weight,'valid_dino_rows':int(good.sum()),
                   'blended_rows':int(good.sum()) if len(base)>=2 and good.all() else 0,
                   'baseline_only_rows':int(len(base)-(good.sum() if len(base)>=2 and good.all() else 0)),
                   'rank_basis':'All three folds required for every case; any incomplete coverage preserves the entire baseline'}
