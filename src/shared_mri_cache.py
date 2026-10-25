"""Bounded local cache of rendered uint8 slices, before model-specific TTA/heads."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np

def digest_file(path):
    if not path: return None
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def emit(event,**fields):
    folder=os.environ.get('RSNA_CACHE_LOG')
    if folder:
        try:
            p=Path(folder); p.mkdir(parents=True,exist_ok=True)
            with (p/(str(os.getpid())+'.jsonl')).open('a') as f:
                f.write(json.dumps(dict(event=event,mode=os.environ.get('RSNA_CACHE_MODE'),**fields))+'\n')
        except OSError: pass

def identity(ds):
    keys=['data','split','size','eval_stride','views','slices','phys_mm','side_threshold','side_max_x','side_position_mm','extent_meta','fluid']
    config={k:getattr(ds,k) for k in keys}
    config['anchor_sha']=digest_file(ds.anchor_weights)
    config['side_sha']=digest_file(ds.side_model)
    here=Path(__file__).parent
    config['source_sha']={n:digest_file(here/n) for n in ['compact_train.py','compact_infer.py','px_cache.py','probe_infer.py','side_common.py','knee_anchor_bank_v2.py']}
    config['helper_sha']=digest_file(__file__)
    native=Path(ds.code)/'training'
    config['native_sha']={p.name:digest_file(p) for p in sorted(native.glob('*.py'))}
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()

def location(ds,uid):
    if not hasattr(ds,'_cache_identity'): ds._cache_identity=identity(ds)
    root=Path(os.environ['RSNA_SHARED_CACHE'])
    folder=root/ds._cache_identity
    folder.mkdir(parents=True,exist_ok=True)
    return root,folder/(hashlib.sha256(uid.encode()).hexdigest()+'.npz')

def save_item(ds,uid,px,indices,out):
    if os.environ.get('RSNA_CACHE_MODE')!='write' or not os.environ.get('RSNA_SHARED_CACHE'): return
    try:
        import fcntl
        root,path=location(ds,uid)
        # Store each referenced rendered slice once, not three copies per window.
        unique,inverse=np.unique(indices,return_inverse=True)
        pixels=np.asarray(px[unique],dtype=np.uint8)
        index=inverse.reshape(indices.shape).astype(np.int32)
        meta=out['meta'].numpy()
        info={k:out[k] for k in ['uid','anchor','side','fluid'] if k in out}
        text=json.dumps(info,sort_keys=True)
        need=pixels.nbytes+index.nbytes+meta.nbytes+len(text.encode())*4+16384
        cap=int(os.environ.get('RSNA_CACHE_MAX_GIB','32'))*1024**3
        reserve=int(os.environ.get('RSNA_CACHE_RESERVE_GIB','6'))*1024**3
        with (root/'allocation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            used=sum(p.stat().st_size for p in root.glob('*/*.npz'))
            if used+need>cap or shutil.disk_usage(root).free<reserve+need:
                emit('write_skipped_space',uid=uid); return
            tmp=path.with_suffix('.tmp.npz')
            try:
                np.savez(tmp,pixels=pixels,indices=index,meta=meta,info=np.asarray(text),fingerprint=np.asarray(ds._cache_identity))
                tmp.replace(path)
            finally:
                if tmp.exists(): tmp.unlink()
        emit('write',uid=uid,bytes=path.stat().st_size,fingerprint=ds._cache_identity)
    except Exception as exc:
        emit('write_error',uid=uid,error=repr(exc))

def load_item(ds,uid):
    import torch
    started=time.perf_counter()
    root,path=location(ds,uid)
    if not path.is_file(): emit('miss',uid=uid,fingerprint=ds._cache_identity); return None
    try:
        with np.load(path,allow_pickle=False) as z:
            assert str(z['fingerprint'])==ds._cache_identity
            info=json.loads(str(z['info'])); assert info['uid']==uid
            px=z['pixels']; ix=z['indices']; meta=z['meta']
            assert px.dtype==np.uint8 and px.shape[1:]==(ds.size,ds.size)
            assert ix.ndim==2 and ix.shape[1]==ds.slices and len(ix)>0
            assert ix.min()>=0 and ix.max()<len(px)
            assert meta.shape==(len(ix),9 if ds.extent_meta else 6) and np.isfinite(meta).all()
            images=np.stack([px[ix[:,i]] for i in range(ds.slices)],axis=1)
        out=dict(info,error=None,images=torch.from_numpy(images),meta=torch.from_numpy(meta),
                 seconds=[0.,round(time.perf_counter()-started,3),0.])
        if ds.fluid: assert 'fluid' in out; out['fluid_s']=0.
        emit('hit',uid=uid,bytes=path.stat().st_size)
        return out
    except Exception as exc:
        emit('read_error',uid=uid,error=repr(exc)); return None
    finally:
        # This generated entry is single-use; R7 workers have disjoint assignments.
        try: path.unlink(missing_ok=True)
        except OSError: pass

def install(cls):
    original=cls.__getitem__
    def getitem(ds,i):
        if os.environ.get('RSNA_CACHE_MODE')!='read' or getattr(ds,'_cache_disabled',False):
            return original(ds,i)
        uid=ds.uids[i]
        try: cached=load_item(ds,uid)
        except Exception as exc:
            emit('cache_error',uid=uid,error=repr(exc)); cached=None
        if cached is None: return original(ds,i)
        if not getattr(ds,'_cache_checked',False):
            import torch
            fresh=original(ds,i)
            same=(fresh.get('error') is None and torch.equal(fresh['images'],cached['images'])
                  and torch.equal(fresh['meta'],cached['meta'])
                  and fresh.get('fluid')==cached.get('fluid')
                  and fresh.get('side')==cached.get('side')
                  and fresh.get('anchor')==cached.get('anchor'))
            ds._cache_checked=True
            if not same:
                ds._cache_disabled=True
                emit('verification_failed',uid=uid)
                return fresh
            emit('verification_passed',uid=uid)
        return cached
    cls.__getitem__=getitem
