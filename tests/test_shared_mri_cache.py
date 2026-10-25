import json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import shared_mri_cache as C

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        (self.root/'training').mkdir()
        self.ds=types.SimpleNamespace(data='/data',split='test',size=4,eval_stride=1,views='v4',slices=3,phys_mm=3.5,
            side_threshold=2.,side_max_x=0.,side_position_mm=20.,extent_meta=True,fluid=True,
            anchor_weights=None,side_model=None,code=str(self.root))
        self.px=np.arange(6*4*4,dtype=np.uint8).reshape(6,4,4)
        self.ix=np.array([[0,1,2],[1,2,3],[3,4,5]])
        self.out=dict(uid='one',meta=torch.zeros(3,9),anchor={'method':'test'},side={'side':'L'},fluid={'eff':1.5})
        self.env=patch.dict(os.environ,RSNA_SHARED_CACHE=str(self.root/'cache'),RSNA_CACHE_LOG=str(self.root/'logs'),RSNA_CACHE_MODE='write',RSNA_CACHE_MAX_GIB='1',RSNA_CACHE_RESERVE_GIB='0')
        self.env.start()
        self.hashpatch=patch.object(C,'digest_file',return_value='same'); self.hashpatch.start()
        self.flock=patch.dict(sys.modules,fcntl=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1)); self.flock.start()
    def tearDown(self):
        self.flock.stop(); self.hashpatch.stop(); self.env.stop(); self.tmp.cleanup()
    def test_exact_roundtrip_and_consumption(self):
        C.save_item(self.ds,'one',self.px,self.ix,self.out)
        os.environ['RSNA_CACHE_MODE']='read'
        result=C.load_item(self.ds,'one')
        expected=np.stack([self.px[self.ix[:,j]] for j in range(3)],1)
        np.testing.assert_array_equal(result['images'].numpy(),expected)
        torch.testing.assert_close(result['meta'],self.out['meta'])
        self.assertEqual(result['fluid'],self.out['fluid'])
        self.assertIsNone(C.load_item(self.ds,'one'))
    def test_config_change_misses(self):
        C.save_item(self.ds,'one',self.px,self.ix,self.out)
        changed=types.SimpleNamespace(**{k:v for k,v in vars(self.ds).items() if not k.startswith('_')})
        changed.eval_stride=3
        self.assertIsNone(C.load_item(changed,'one'))
    def test_capacity_skip(self):
        os.environ['RSNA_CACHE_MAX_GIB']='0'
        C.save_item(self.ds,'one',self.px,self.ix,self.out)
        self.assertIsNone(C.load_item(self.ds,'one'))
    def test_corruption_falls_back(self):
        C.save_item(self.ds,'one',self.px,self.ix,self.out)
        _,p=C.location(self.ds,'one'); p.write_bytes(b'bad')
        self.assertIsNone(C.load_item(self.ds,'one'))
    def test_verification_mismatch_returns_fresh(self):
        class DS:
            def __getitem__(self,i): return dict(error=None,images=torch.ones(1),meta=torch.ones(1),fluid=None)
        ds=DS(); ds.uids=['one']
        C.install(DS); os.environ['RSNA_CACHE_MODE']='read'
        with patch.object(C,'load_item',return_value=dict(error=None,images=torch.zeros(1),meta=torch.ones(1),fluid=None)):
            out=ds[0]
            self.assertEqual(out['images'].item(),1)
            self.assertTrue(ds._cache_disabled)

if __name__=='__main__': unittest.main(verbosity=2)
