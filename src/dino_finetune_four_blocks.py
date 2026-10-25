"""Controlled last-four-block fine-tuning from the frozen-feature experiment."""
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from dino_branch_train import ARCH, UID, LABELS, Head, Images, digest, loss_fn, metrics


def unfreeze_tail(encoder):
    encoder.requires_grad_(False)
    assert len(encoder.blocks) >= 4
    for block in encoder.blocks[-4:]: block.requires_grad_(True)
    encoder.norm.requires_grad_(True)
    return [n for n,p in encoder.named_parameters() if p.requires_grad]


class Model(nn.Module):
    def __init__(self,encoder,head,mean,std):
        super().__init__(); self.encoder=encoder;self.head=head
        self.register_buffer('mean',torch.tensor(mean)[None,:,None,None])
        self.register_buffer('std',torch.tensor(std)[None,:,None,None])
    def train(self,mode=True):
        super().train(mode)
        # Keep encoder stochastic layers disabled; gradients still flow through its tail.
        self.encoder.eval()
        return self
    def forward(self,x,mask):
        b=x.shape[0]
        x=x.reshape(-1,1,224,224).float().div(255).expand(-1,3,-1,-1)
        x=(x-self.mean)/self.std
        with torch.autocast(device_type=x.device.type,dtype=torch.float16,enabled=x.is_cuda):
            f=self.encoder(x)
        # Same FP16 feature rounding and FP32 head as v1; cast remains differentiable.
        f=f.half().float().reshape(b,3,12,-1)
        return self.head(f,mask)


class Studies(Images):
    def __init__(self,df,cache):
        self.df=df.reset_index(drop=True)
        super().__init__(self.df[UID],cache)
    def __getitem__(self,i):
        x,m,uid,h=super().__getitem__(i)
        return x,m,torch.tensor(self.df.loc[i,LABELS].to_numpy(dtype='float32'))


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();ys=[];ps=[];t=time.time()
    for i,(x,m,y) in enumerate(loader):
        ps.append(model(x.to(device),m.to(device)).sigmoid().cpu().numpy());ys.append(y.numpy())
        if (i+1)%100==0: print(f'EVAL batches {i+1}/{len(loader)} elapsed {(time.time()-t)/60:.1f} min',flush=True)
    y,p=np.concatenate(ys),np.concatenate(ps)
    return metrics(y,p),p


def save_predictions(rows,p,path):
    f=rows[[UID,'group']+LABELS].reset_index(drop=True).rename(columns={c:'target::'+c for c in LABELS})
    for j,c in enumerate(LABELS):f['pred::'+c]=p[:,j]
    f.to_csv(path,index=False)


def main():
    import timm
    ap=argparse.ArgumentParser()
    for name in ['base','cache','out']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=6);args=ap.parse_args()
    assert torch.cuda.is_available(),'Enable GPU'
    if args.out.exists() and any(args.out.iterdir()):raise RuntimeError('Use an empty output directory; do not mix runs')
    args.out.mkdir(parents=True,exist_ok=True);out=args.out;start=time.time()
    seed=20261020;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
    cfg=json.loads((args.base/'config.json').read_text());fcfg=cfg['feature_config']
    assert fcfg['arch']==ARCH and fcfg['size']==224
    assert timm.__version__==fcfg['timm'],'Use same timm version as v1'
    for f,key in [('split.csv','split_sha256')]:assert digest(args.base/f)==cfg[key]
    assert digest(args.base/'dino_encoder.pt')==fcfg['encoder_sha256']
    assert digest(args.cache/'config.json')==fcfg['cache_config_sha256']
    split=pd.read_csv(args.base/'split.csv',dtype={UID:str})
    assert split[UID].is_unique
    active=split[split.split.ne('excluded')]
    assert active.groupby('group').split.nunique().max()==1
    for uid in active[UID]:assert (args.cache/(uid+'.npz')).is_file(),uid
    shutil.copyfile(args.base/'split.csv',out/'split.csv')
    encoder=timm.create_model(ARCH,pretrained=False,num_classes=0,img_size=224)
    encoder.load_state_dict(torch.load(args.base/'dino_encoder.pt',weights_only=True,map_location='cpu'),strict=True)
    head=Head();old=torch.load(args.base/'best_head.pt',weights_only=True,map_location='cpu')
    assert old['labels']==LABELS and old['feature_signature']==fcfg['signature']
    head.load_state_dict(old['state_dict'],strict=True)
    names=unfreeze_tail(encoder)
    raw=Model(encoder,head,fcfg['mean'],fcfg['std']).cuda()
    model=nn.DataParallel(raw) if torch.cuda.device_count()>1 else raw
    dfs={s:split[split.split.eq(s)].copy() for s in ['train','dev','holdout','gold_reference']}
    loaders={s:DataLoader(Studies(df,args.cache),batch_size=2,shuffle=s=='train',num_workers=2,pin_memory=True) for s,df in dfs.items()}
    audit={'seed':seed,'epochs_max':args.epochs,'encoder_lr':1e-5,'head_lr':1e-4,'weight_decay':.01,
           'unfrozen_blocks':4,'unfrozen_encoder_parameters':names,'batch_studies':2,'gradient_accumulation':4,
           'input_recipe':'same 36 slices as v1; no new augmentation or crop','selection':'dev only; epoch zero eligible',
           'holdout_previously_inspected':True,'gold_independent_test':False,'baseline_complementarity_tested':False,
           'base_config':cfg,'base_head_sha256':digest(args.base/'best_head.pt'),'script_sha256':digest(__file__),
           'helper_sha256':digest(Path(__file__).with_name('dino_branch_train.py'))}
    (out/'config.json').write_text(json.dumps(audit,indent=2))
    print('BASELINE reconstruction on dev',flush=True)
    baseline,p=evaluate(model,loaders['dev'],'cuda')
    previous=pd.read_csv(args.base/'dev_predictions.csv',dtype={UID:str}).set_index(UID)
    expected=previous.loc[dfs['dev'][UID],['pred::'+c for c in LABELS]].to_numpy()
    delta=float(np.max(np.abs(p-expected)))
    original=json.loads((args.base/'summary.json').read_text())['metrics']['dev']['macro_auc']
    check={'dev':baseline,'max_probability_difference':delta,'original_dev_auc':original}
    (out/'baseline_check.json').write_text(json.dumps(check,indent=2))
    assert delta<.003 and abs(baseline['macro_auc']-original)<.002,'Baseline mismatch: inspect baseline_check.json; training stopped'
    save_predictions(dfs['dev'],p,out/'baseline_dev_predictions.csv')
    best=baseline['macro_auc'];best_epoch=0;stale=0;history=[]
    def save(epoch):
        # Detached CPU copies avoid a large GPU checkpoint allocation.
        torch.save({'encoder':{k:v.detach().cpu() for k,v in raw.encoder.state_dict().items()},
                    'head':{k:v.detach().cpu() for k,v in raw.head.state_dict().items()},
                    'epoch':epoch,'labels':LABELS,'feature_config':fcfg},out/'best_model.pt')
    save(0)
    opt=torch.optim.AdamW([{'params':[p for p in raw.encoder.parameters() if p.requires_grad],'lr':1e-5},
                          {'params':raw.head.parameters(),'lr':1e-4}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=0.)
    scaler=torch.amp.GradScaler('cuda');limit=7.5*3600
    for epoch in range(1,args.epochs+1):
        # Leave room for final evaluation and packaging; never start an obviously over-budget epoch.
        if history and time.time()-start+history[-1]['seconds']+900>limit:break
        t=time.time();model.train();opt.zero_grad(set_to_none=True);losses=[];aborted=False
        for i,(x,m,y) in enumerate(loaders['train']):
            if time.time()-start>limit:aborted=True;break
            n=len(loaders['train']);group_size=min(4,n-(i//4)*4)
            loss=loss_fn(model(x.cuda(),m.cuda()),y.cuda())
            assert torch.isfinite(loss),'Nonfinite loss'
            scaler.scale(loss/group_size).backward();losses.append(float(loss.detach()))
            if (i+1)%4==0 or i+1==n:
                scaler.unscale_(opt);nn.utils.clip_grad_norm_([p for p in raw.parameters() if p.requires_grad],1.)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            if (i+1)%100==0:
                elapsed=time.time()-t
                print(f'EPOCH {epoch} batches {i+1}/{n} loss {np.mean(losses):.4f} ETA {(n-i-1)*elapsed/(i+1)/60:.1f} min',flush=True)
        if aborted:
            print('TIME BUDGET: incomplete epoch discarded; restoring best checkpoint',flush=True);break
        score,p=evaluate(model,loaders['dev'],'cuda');scheduler.step()
        rec={'epoch':epoch,'loss':float(np.mean(losses)),'dev':score,'seconds':time.time()-t}
        history.append(rec)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps(rec),flush=True)
        if score['macro_auc']>best:
            best=score['macro_auc'];best_epoch=epoch;stale=0;save(epoch)
        else:stale+=1
        if stale>=3:break
    ck=torch.load(out/'best_model.pt',weights_only=True,map_location='cpu')
    raw.encoder.load_state_dict(ck['encoder']);raw.head.load_state_dict(ck['head']);del ck
    scores={}
    # No holdout or gold measurements were used for model selection.
    for s in ['dev','holdout','gold_reference']:
        scores[s],p=evaluate(model,loaders[s],'cuda');save_predictions(dfs[s],p,out/(s+'_predictions.csv'))
    result={'status':'complete','selected_epoch':best_epoch,'epochs_completed':len(history),'baseline_dev_auc':baseline['macro_auc'],
            'dev_gain':best-baseline['macro_auc'],'metrics':scores,'counts':{s:len(d) for s,d in dfs.items()},
            'elapsed_seconds':time.time()-start,'submission_ready':False,'holdout_previously_inspected':True}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    import zipfile
    with zipfile.ZipFile(out.parent/'dino-four-block-results-small.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.suffix in ['.json','.jsonl','.csv']:z.write(p,p.name)
    print('RESULTS READY: dino-four-block-results-small.zip',flush=True)


if __name__=='__main__':main()
