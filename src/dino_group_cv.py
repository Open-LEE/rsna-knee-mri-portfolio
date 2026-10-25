"""Three-fold grouped weak-label CV. Fixed schedules, no outer-fold selection."""
import argparse
import gc
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from dino_branch_train import UID,LABELS,ARCH,Head,Features,loss_fn,metrics,digest,evaluate as evaluate_features
from dino_finetune_four_blocks import Model,Studies,unfreeze_tail,evaluate,save_predictions


def assign_folds(frame,seed=20261022,nfold=3):
    out=frame[frame.split.isin(['train','dev','holdout'])].copy().reset_index(drop=True)
    assert out[UID].notna().all() and out[UID].is_unique
    assert out.group.notna().all() and out[LABELS].notna().any(axis=1).all()
    groups=out.groupby('group',sort=True).size()
    assert len(groups)>=nfold
    rng=np.random.default_rng(seed)
    # Random tie order, then greedily balance group sizes without consulting labels.
    order=list(groups.index[rng.permutation(len(groups))]);order.sort(key=lambda g:-groups[g])
    totals=[0]*nfold;mapping={}
    for g in order:
        f=int(np.argmin(totals));mapping[g]=f;totals[f]+=int(groups[g])
    out['fold']=out.group.map(mapping).astype(int)
    assert out.groupby('group').fold.nunique().max()==1
    return out


def package(out):
    with zipfile.ZipFile(out.parent/'dino-group-cv-results-small.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix in ['.csv','.json','.jsonl']:z.write(p,p.relative_to(out))


def main():
    import timm
    ap=argparse.ArgumentParser()
    for k in ['base','cache','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();assert torch.cuda.is_available()
    assert not a.out.exists() or not any(a.out.iterdir()),'Use a new empty output directory'
    a.out.mkdir(parents=True,exist_ok=True);start=time.time();torch.set_num_threads(4)
    source=json.loads((a.base/'config.json').read_text());fc=source['feature_config']
    assert source['backbone_frozen'] is True,'Must use v1 generic encoder, not fine-tuned checkpoint'
    assert timm.__version__==fc['timm'] and fc['arch']==ARCH
    assert digest(a.base/'dino_encoder.pt')==fc['encoder_sha256']
    assert digest(a.base/'split.csv')==source['split_sha256']
    assert digest(a.cache/'config.json')==fc['cache_config_sha256']
    frame=pd.read_csv(a.base/'split.csv',dtype={UID:str});folds=assign_folds(frame)
    assert not set(folds.group)&set(frame.loc[frame.split.eq('gold_reference'),'group'])
    folds.to_csv(a.out/'folds.csv',index=False)
    for uid in folds[UID]:
        assert (a.cache/(uid+'.npz')).is_file(),uid
        with np.load(a.base/'features'/(uid+'.npz'),allow_pickle=False) as z:
            assert str(z['signature'])==fc['signature'] and z['f'].shape==(3,12,384)
            assert np.isfinite(z['f']).all()
    config={'seed':20261022,'nfold':3,'head_epochs':20,'finetune_epochs':6,'unfrozen_blocks':4,
            'head_warmup_lr':.001,'finetune_head_lr':1e-4,'encoder_lr':1e-5,
            'selection':'fixed final epoch, no outer fold early stopping or checkpoint selection',
            'head_initialization':'new random head per fold; never load v1 best_head.pt',
            'encoder_initialization':'same generic pretrained v1 encoder per fold, never a knee-finetuned checkpoint',
            'excluded':'gold_reference and excluded rows from v1',
            'limitations':['weak labels, not clinical ground truth','grouping is exact report/image, not patient identity',
                          'dataset and model family already used for development; not a pristine final test',
                          'cannot establish complementarity with R6/R7 without their matched OOF'],
            'feature_config':fc,'folds_sha256':digest(a.out/'folds.csv'),'script_sha256':digest(__file__)}
    (a.out/'config.json').write_text(json.dumps(config,indent=2))
    print('FOLD COUNTS',folds.fold.value_counts().sort_index().to_dict(),flush=True)
    results=[];all_frozen=[];all_fine=[]
    for fold in range(3):
        t=time.time();seed=20261022+fold
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        train=folds[folds.fold.ne(fold)].copy();val=folds[folds.fold.eq(fold)].copy()
        assert set(train.group).isdisjoint(val.group) and set(train[UID]).isdisjoint(val[UID])
        folder=a.out/f'fold{fold}';folder.mkdir()
        train[[UID,'group','fold']].to_csv(folder/'training_ids.csv',index=False)
        val[[UID,'group','fold']].to_csv(folder/'validation_ids.csv',index=False)
        head=Head().cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        fl=DataLoader(Features(train,a.base/'features'),batch_size=64,shuffle=True,num_workers=2)
        for epoch in range(20):
            head.train();losses=[]
            for f,m,y in fl:
                opt.zero_grad(set_to_none=True);loss=loss_fn(head(f.cuda(),m.cuda()),y.cuda())
                loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step();losses.append(loss.item())
            print(f'FOLD {fold} HEAD {epoch+1}/20 train_loss={np.mean(losses):.5f}',flush=True)
        vl=DataLoader(Features(val,a.base/'features'),batch_size=64,num_workers=2)
        frozen_score,p=evaluate_features(head,vl,'cuda')
        save_predictions(val,p,folder/'frozen_predictions.csv')
        torch.save({'state_dict':head.state_dict(),'labels':LABELS,'fold':fold},folder/'frozen_head.pt')
        del opt,fl,vl
        encoder=timm.create_model(ARCH,pretrained=False,num_classes=0,img_size=224)
        encoder.load_state_dict(torch.load(a.base/'dino_encoder.pt',map_location='cpu',weights_only=True),strict=True)
        unfreeze_tail(encoder)
        raw=Model(encoder,head,fc['mean'],fc['std']).cuda()
        model=nn.DataParallel(raw) if torch.cuda.device_count()>1 else raw
        loader=DataLoader(Studies(train,a.cache),batch_size=2,shuffle=True,num_workers=2,pin_memory=True)
        opt=torch.optim.AdamW([{'params':[p for p in raw.encoder.parameters() if p.requires_grad],'lr':1e-5},
                              {'params':raw.head.parameters(),'lr':1e-4}],weight_decay=.01)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=6)
        scaler=torch.amp.GradScaler('cuda')
        for epoch in range(6):
            model.train();opt.zero_grad(set_to_none=True);losses=[];ep_start=time.time()
            for i,(x,m,y) in enumerate(loader):
                loss=loss_fn(model(x.cuda(),m.cuda()),y.cuda());assert torch.isfinite(loss)
                group_size=min(4,len(loader)-(i//4)*4)
                scaler.scale(loss/group_size).backward();losses.append(float(loss.detach()))
                if (i+1)%4==0 or i+1==len(loader):
                    scaler.unscale_(opt);nn.utils.clip_grad_norm_([p for p in raw.parameters() if p.requires_grad],1.)
                    scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
                if (i+1)%100==0:
                    print(f'FOLD {fold} FINETUNE {epoch+1}/6 batch {i+1}/{len(loader)} elapsed={(time.time()-ep_start)/60:.1f} min',flush=True)
            sched.step()
            rec={'fold':fold,'epoch':epoch+1,'train_loss':float(np.mean(losses)),'seconds':time.time()-ep_start}
            with (folder/'train_metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        # Only now evaluate the outer fold: no outer labels used for optimization or epoch selection.
        valid_loader=DataLoader(Studies(val,a.cache),batch_size=2,num_workers=2,pin_memory=True)
        fine_score,p=evaluate(model,valid_loader,'cuda');save_predictions(val,p,folder/'finetuned_predictions.csv')
        torch.save({'encoder':{k:v.detach().cpu() for k,v in raw.encoder.state_dict().items()},
                    'head':{k:v.detach().cpu() for k,v in raw.head.state_dict().items()},
                    'fold':fold,'epoch':6,'labels':LABELS,'feature_config':fc},folder/'final_model.pt')
        result={'fold':fold,'train':len(train),'validation':len(val),'frozen':frozen_score,'finetuned':fine_score,'seconds':time.time()-t}
        (folder/'summary.json').write_text(json.dumps(result,indent=2));results.append(result)
        print('FOLD COMPLETE',json.dumps(result),flush=True)
        for name,collector in [('frozen',all_frozen),('finetuned',all_fine)]:
            table=pd.read_csv(folder/(name+'_predictions.csv'),dtype={UID:str});table['fold']=fold;collector.append(table)
        (a.out/'progress.json').write_text(json.dumps({'completed_folds':len(results),'required_folds':3},indent=2));package(a.out)
        del model,raw,head,encoder,opt,sched,scaler,loader,valid_loader
        gc.collect();torch.cuda.empty_cache()
    summary={'status':'complete','folds':results,'submission_ready':False,'elapsed_seconds':time.time()-start,'pooled_oof':{}}
    for name,tables in [('frozen',all_frozen),('finetuned',all_fine)]:
        df=pd.concat(tables,ignore_index=True);assert df[UID].is_unique and set(df[UID])==set(folds[UID])
        df.to_csv(a.out/(name+'_oof.csv'),index=False)
        summary['pooled_oof'][name]=metrics(df[['target::'+c for c in LABELS]].to_numpy(),df[['pred::'+c for c in LABELS]].to_numpy())
    summary['mean_fold_macro_auc']={name:float(np.mean([r[name]['macro_auc'] for r in results])) for name in ['frozen','finetuned']}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));package(a.out)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
