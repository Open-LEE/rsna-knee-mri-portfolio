import json
from pathlib import Path
import numpy as np


def compare(root):
    root=Path(root);runs={};receipts={}
    for name in ['v025_a','v025_b','v029']:
        with np.load(root/name/'predictions.npz',allow_pickle=False) as z:runs[name]={k:z[k].copy() for k in z.files}
        receipts[name]=json.loads((root/name/'receipt.json').read_text())
    a=runs['v025_a']
    for name,r in runs.items():
        assert np.array_equal(a['uids'],r['uids']) and np.array_equal(a['labels'],r['labels'])
        assert np.isfinite(r['prob']).all() and ((r['prob']>=0)&(r['prob']<=1)).all()
        assert np.isfinite(r['features']).all()
        for key in ['checkpoint_sha256','input_sha256','torch','gpu']:
            assert receipts[name][key]==receipts['v025_a'][key],key
    noise=np.abs(a['prob']-runs['v025_b']['prob'])
    diff=np.abs(a['prob']-runs['v029']['prob'])
    per={str(c):float(diff[:,j].max()) for j,c in enumerate(a['labels'])}
    # A diagnostic threshold, not an AUC or equivalence guarantee.
    threshold=max(.001,10*float(noise.max()))
    summary={'cases':len(a['uids']),'repeat_025_max_probability_difference':float(noise.max()),
             'cross_version_max_probability_difference':float(diff.max()),'cross_version_mean_probability_difference':float(diff.mean()),
             'cross_version_max_feature_difference':float(np.abs(a['features']-runs['v029']['features']).max()),
             'per_class_max_probability_difference':per,'diagnostic_threshold':threshold,
             'repeat_noise_above_0_001':bool(noise.max()>.001),
             'difference_above_threshold':bool(diff.max()>threshold),
             'conclusion_scope':'Only sampled inputs under this runtime; does not explain hidden-test score or prove equivalence on all cases',
             'no_training_no_auc_no_submission':True}
    (root/'comparison.json').write_text(json.dumps(summary,indent=2));return summary
