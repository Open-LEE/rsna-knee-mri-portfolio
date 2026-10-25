import unittest
import torch
from torch import nn
from dino_branch_train import Head,loss_fn
from dino_finetune import Model,unfreeze_tail


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Linear(3,384)
        self.blocks=nn.ModuleList([nn.Linear(384,384) for _ in range(4)])
        self.norm=nn.LayerNorm(384)
    def forward(self,x):
        f=self.stem(x.mean((2,3)))
        for block in self.blocks:f=torch.tanh(block(f))
        return self.norm(f)


class FineTuneTests(unittest.TestCase):
    def test_gradients_reach_only_tail_and_head(self):
        torch.set_num_threads(2);e=FakeEncoder();unfreeze_tail(e)
        m=Model(e,Head(),[.485,.456,.406],[.229,.224,.225]);m.train()
        self.assertFalse(e.training);self.assertTrue(m.head.training)
        x=torch.randint(0,256,(1,3,12,224,224),dtype=torch.uint8);mask=torch.ones(1,3,dtype=torch.bool)
        loss_fn(m(x,mask),torch.zeros(1,12)).backward()
        self.assertIsNone(e.stem.weight.grad);self.assertIsNone(e.blocks[1].weight.grad)
        for p in [e.blocks[2].weight,e.blocks[3].weight,e.norm.weight,m.head.value.weight]:
            self.assertIsNotNone(p.grad);self.assertGreater(p.grad.abs().sum().item(),0)

    def test_baseline_same_feature_recipe(self):
        torch.manual_seed(3);e=FakeEncoder();h=Head();m=Model(e,h,[.485,.456,.406],[.229,.224,.225]).eval()
        x=torch.randint(0,256,(1,3,12,224,224),dtype=torch.uint8);mask=torch.tensor([[True,False,True]])
        with torch.no_grad():
            rgb=x.reshape(-1,1,224,224).float().div(255).expand(-1,3,-1,-1)
            f=e((rgb-m.mean)/m.std).half().float().reshape(1,3,12,384)
            torch.testing.assert_close(m(x,mask),h(f,mask))


if __name__=='__main__':unittest.main()
