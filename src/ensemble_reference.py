"""Complete-coverage rank fusion reference; not the full competition inference."""
import numpy as np
import pandas as pd

from dino_branch_train import UID, LABELS


def blend_complete(coat384, coat224, r6, r7):
    frames = [coat384, coat224, r6, r7]
    ids = coat384[UID]
    assert len(ids) and ids.notna().all() and ids.is_unique
    ranks = []
    for frame in frames:
        assert set(frame.columns) == {UID, *LABELS}
        assert frame[UID].notna().all() and frame[UID].is_unique
        assert set(frame[UID]) == set(ids), 'Complete identical case coverage required'
        values = frame.set_index(UID).loc[ids, LABELS].astype(float)
        assert np.isfinite(values.to_numpy()).all()
        ranks.append(values.rank(method='average', pct=True).to_numpy())
    weights = np.array([{'Lateral OA': .2, 'PF OA': .6,
                         'Synovitis': .8, "Baker's": .35}.get(c, .5) for c in LABELS])
    scores = weights * (.65*ranks[0] + .35*ranks[1]) + (1-weights)*(.8*ranks[2] + .2*ranks[3])
    out = pd.DataFrame(scores, columns=LABELS)
    out.insert(0, UID, ids.to_numpy())
    return out
