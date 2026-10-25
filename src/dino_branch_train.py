"""Frozen DINOv2 MRI branch: grouped development and one-time weak holdout."""
import argparse
import hashlib
import json
import random
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

UID = 'StudyInstanceUID'
LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
ARCH = 'vit_small_patch14_dinov2.lvd142m'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def report_hash(s):
    s = ' '.join(unicodedata.normalize('NFKC', str(s)).casefold().split())
    return hashlib.sha256(s.encode()).hexdigest() if s else ''


def make_split(official, weak, image_hashes, seed=20261019):
    from sklearn.model_selection import GroupShuffleSplit
    for df in [official, weak]:
        assert df[UID].notna().all() and df[UID].is_unique, 'Invalid study IDs'
    table = official[[UID,'Report']+LABELS].copy().reset_index(drop=True)
    gold = table[LABELS].notna().all(1)
    empty = table[LABELS].isna().all(1)
    assert (gold | empty).all(), 'Unexpected partially official-labelled rows'
    parent = list(range(len(table)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    reports = table.Report.fillna('').map(report_hash)
    for i, uid in enumerate(table[UID]):
        for key in [('report',reports[i]),('image',image_hashes[uid])]:
            if not key[1]: continue
            if key in seen: parent[find(i)] = find(seen[key])
            else: seen[key] = i
    table['group'] = [find(i) for i in range(len(table))]
    gold_groups = set(table.loc[gold,'group'])
    table['split'] = 'excluded'
    table.loc[gold,'split'] = 'gold_reference'
    eligible = empty & ~table.group.isin(gold_groups) & reports.ne('')
    aligned = weak.set_index(UID).reindex(table[UID])
    for c in LABELS:
        v = aligned[c+'__verdict'].astype('string').str.strip().str.upper()
        assert (v.isna() | v.isin(['YES','NO','UNK'])).all(), c
        vals = v.map({'YES':1.,'NO':0.,'UNK':np.nan}).to_numpy(dtype=float,na_value=np.nan)
        table.loc[empty,c] = vals[empty.to_numpy()]
    eligible &= table[LABELS].notna().any(axis=1)
    ids = np.flatnonzero(eligible)
    a,b = next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=seed).split(ids,groups=table.loc[ids,'group']))
    pool,hold = ids[a],ids[b]
    a,b = next(GroupShuffleSplit(n_splits=1,test_size=.15/.85,random_state=seed+1).split(pool,groups=table.loc[pool,'group']))
    for name, rows in [('train',pool[a]),('dev',pool[b]),('holdout',hold)]: table.loc[rows,'split'] = name
    sets = [set(table.loc[table.split.eq(s),'group']) for s in ['train','dev','holdout','gold_reference']]
    assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
    return table.drop(columns='Report')


class Images(Dataset):
    def __init__(self, ids, cache): self.ids,self.cache = list(ids),Path(cache)
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        uid = self.ids[i]
        with np.load(self.cache/(uid+'.npz'),allow_pickle=False) as z:
            x,m = z['x'].copy(),z['mask'].astype(bool)
        assert x.shape==(3,12,224,224) and x.dtype==np.uint8, (uid,x.shape,x.dtype)
        assert m.shape==(3,) and m.any(), uid
        h=hashlib.sha256(x.tobytes()+m.tobytes()).hexdigest()
        return torch.from_numpy(x),torch.from_numpy(m),uid,h


def extract(ids,cache,out,device):
    import timm
    fdir=out/'features'; fdir.mkdir(exist_ok=True)
    model=timm.create_model(ARCH,pretrained=True,num_classes=0,img_size=224).eval().to(device)
    cfg=timm.data.resolve_model_data_config(model)
    torch.save(model.cpu().state_dict(),out/'dino_encoder.pt'); model.to(device)
    config={'arch':ARCH,'size':224,'slices':12,'planes':3,'mean':list(cfg['mean']),'std':list(cfg['std']),
            'encoder_sha256':digest(out/'dino_encoder.pt'),'cache_config_sha256':digest(cache/'config.json'),
            'timm':timm.__version__,'torch':torch.__version__,'input':'uint8 / 255; grayscale repeated RGB; no spatial crop'}
    # Never silently reuse features from a different encoder or preprocessing recipe.
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config['signature']=signature
    (out/'feature_config.json').write_text(json.dumps(config,indent=2))
    hashes={}; todo=[]
    for uid in ids:
        p=fdir/(uid+'.npz')
        if p.exists():
            with np.load(p,allow_pickle=False) as z:
                if str(z['signature'])==signature:
                    hashes[uid]=str(z['image_hash']); continue
        todo.append(uid)
    mean=torch.tensor(cfg['mean'],device=device)[None,:,None,None]
    std=torch.tensor(cfg['std'],device=device)[None,:,None,None]
    if torch.cuda.device_count()>1: model=nn.DataParallel(model)
    loader=DataLoader(Images(todo,cache),batch_size=4,num_workers=2,pin_memory=True)
    start=time.time(); done=0
    with torch.inference_mode():
        for x,m,uids,hh in loader:
            b=len(uids)
            images=x.to(device).reshape(-1,1,224,224).float().div_(255).expand(-1,3,-1,-1)
            images=(images-mean)/std
            with torch.autocast('cuda',dtype=torch.float16): f=model(images)
            f=f.float().reshape(b,3,12,-1).cpu().numpy().astype('float16')
            assert np.isfinite(f).all()
            for j,uid in enumerate(uids):
                p=fdir/(uid+'.npz'); tmp=p.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,f=f[j],mask=m[j].numpy(),image_hash=hh[j],signature=signature)
                tmp.replace(p);hashes[uid]=hh[j]
            done+=b
            if done%40==0 or done==len(todo):
                elapsed=time.time()-start
                print(f'FEATURES {done}/{len(todo)} elapsed={elapsed/60:.1f} min ETA={(len(todo)-done)*elapsed/max(done,1)/60:.1f} min',flush=True)
    del model;torch.cuda.empty_cache()
    return hashes,config


class Features(Dataset):
    def __init__(self,df,folder):
        self.df=df.reset_index(drop=True); self.folder=folder
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]
        with np.load(self.folder/(row[UID]+'.npz'),allow_pickle=False) as z:
            f=torch.from_numpy(z['f'].astype('float32'));m=torch.from_numpy(z['mask'].copy())
        y=torch.tensor(row[LABELS].to_numpy(dtype='float32'))
        return f,m,y


class Head(nn.Module):
    def __init__(self,dim=384):
        super().__init__()
        self.proj=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,128),nn.GELU())
        self.position=nn.Parameter(torch.zeros(1,3,12,128))
        self.attn=nn.Linear(128,12)
        self.value=nn.Linear(128,12)
        self.bias=nn.Parameter(torch.zeros(12))
        self.drop=nn.Dropout(.2)
    def forward(self,f,mask):
        h=self.proj(f)+self.position
        b=h.shape[0]; h=self.drop(h).reshape(b,36,128)
        good=mask[:,:,None].expand(-1,-1,12).reshape(b,36,1)
        a=self.attn(h).masked_fill(~good,-1e4).softmax(1)
        return (a*self.value(h)).sum(1)+self.bias


def loss_fn(z,y):
    valid=torch.isfinite(y)
    loss=nn.functional.binary_cross_entropy_with_logits(z,torch.nan_to_num(y),reduction='none')
    counts=valid.sum(0); usable=counts>0
    return ((loss*valid).sum(0)/counts.clamp_min(1))[usable].mean()


def metrics(y,p):
    from sklearn.metrics import roc_auc_score
    result={}
    for j,c in enumerate(LABELS):
        m=np.isfinite(y[:,j]); truth=y[m,j]
        result[c]=float(roc_auc_score(truth,p[m,j])) if len(np.unique(truth))==2 else None
    available=[v for v in result.values() if v is not None]
    return {'macro_auc':float(np.mean(available)) if available else None,'valid_classes':len(available),'per_class':result}


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval(); yy=[]; pp=[]
    for f,m,y in loader:
        pp.append(model(f.to(device),m.to(device)).sigmoid().cpu().numpy());yy.append(y.numpy())
    y,p=np.concatenate(yy),np.concatenate(pp)
    return metrics(y,p),p


def main():
    ap=argparse.ArgumentParser()
    for name in ['data','weak','cache','out']: ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=20);a=ap.parse_args()
    assert torch.cuda.is_available(),'Enable GPU'
    seed=20261019;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'summary.json').exists(): raise RuntimeError('Completed experiment exists; use a fresh output directory')
    started=time.time()
    official=pd.read_csv(a.data/'train.csv',dtype={UID:str});weak=pd.read_csv(a.weak,dtype={UID:str})
    assert official[UID].is_unique and official[UID].notna().all()
    for c in LABELS:
        assert official[c].dropna().isin([0,1]).all()
        assert c+'__verdict' in weak
    assert all((a.cache/(u+'.npz')).is_file() for u in official[UID]),'Incomplete cache; mount full 4407-study output'
    hashes,fcfg=extract(official[UID],a.cache,a.out,'cuda')
    split=make_split(official,weak,hashes,seed)
    split.to_csv(a.out/'split.csv',index=False)
    coverage={s:{c:{'positive':int((d[c]==1).sum()),'negative':int((d[c]==0).sum()),'unknown':int(d[c].isna().sum())} for c in LABELS} for s,d in split.groupby('split')}
    (a.out/'label_coverage.json').write_text(json.dumps(coverage,indent=2))
    counts=split.split.value_counts().to_dict();print('SPLITS',counts,flush=True)
    for c,v in coverage['dev'].items():
        if not v['positive'] or not v['negative']: raise RuntimeError(f'Dev lacks both classes: {c}; do not select a different seed based on results')
    config={'seed':seed,'epochs_max':a.epochs,'patience':5,'lr':.001,'weight_decay':.01,'backbone_frozen':True,
            'selection':'dev weak macro AUC only','feature_config':fcfg,'official_sha256':digest(a.data/'train.csv'),
            'weak_sha256':digest(a.weak),'script_sha256':digest(__file__),'split_sha256':digest(a.out/'split.csv'),
            'grouping':'connected components of normalized exact report and exact image hashes; not patient-level IDs',
            'weak_holdout_is_clinical_ground_truth':False,'gold_independent_test':False,'baseline_complementarity_tested':False}
    (a.out/'config.json').write_text(json.dumps(config,indent=2))
    loaders={s:DataLoader(Features(split[split.split.eq(s)],a.out/'features'),batch_size=64,shuffle=s=='train',num_workers=2) for s in ['train','dev','holdout','gold_reference']}
    model=Head().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best=-float('inf'); stale=0
    for epoch in range(1,a.epochs+1):
        model.train();total=0.;steps=0
        for f,m,y in loaders['train']:
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(f.cuda(),m.cuda()),y.cuda())
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=loss.item();steps+=1
        score,_=evaluate(model,loaders['dev'],'cuda')
        record={'epoch':epoch,'loss':total/steps,'dev':score}
        with (a.out/'metrics.jsonl').open('a') as h:h.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if score['macro_auc']>best:
            best=score['macro_auc'];stale=0
            torch.save({'state_dict':model.state_dict(),'epoch':epoch,'labels':LABELS,'feature_signature':fcfg['signature']},a.out/'best_head.pt')
        else:stale+=1
        if stale>=5:break
    ckpt=torch.load(a.out/'best_head.pt',map_location='cuda',weights_only=True);model.load_state_dict(ckpt['state_dict'])
    scores={}
    # The holdout and gold are first evaluated only after the dev-selected model is frozen.
    for name in ['dev','holdout','gold_reference']:
        scores[name],pred=evaluate(model,loaders[name],'cuda')
        rows=split.loc[split.split.eq(name),[UID,'group']+LABELS].reset_index(drop=True)
        rows=rows.rename(columns={c:'target::'+c for c in LABELS})
        for j,c in enumerate(LABELS):rows['pred::'+c]=pred[:,j]
        rows.to_csv(a.out/(name+'_predictions.csv'),index=False)
    result={'status':'training_complete','selected_epoch':ckpt['epoch'],'counts':counts,'metrics':scores,
            'elapsed_seconds':time.time()-started,'submission_ready':False,'complementarity_with_r6_r7_unknown':True}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
