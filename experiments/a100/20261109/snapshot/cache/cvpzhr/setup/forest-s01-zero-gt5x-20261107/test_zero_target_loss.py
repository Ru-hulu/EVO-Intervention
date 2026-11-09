import unittest
import torch
from zero_target_loss import zero_target_weights


class ZeroTargetLoss(unittest.TestCase):
    def gradients(self, multiplier, target=None, valid=None):
        target=torch.tensor([0.,.37,1.,0.]) if target is None else target
        valid=torch.tensor([True,True,True,False]) if valid is None else valid
        logits=torch.linspace(-2,2,404,dtype=torch.float64).reshape(4,101).clone().requires_grad_()
        positions=target.double()*100
        lower=positions.floor().long();upper=(lower+1).clamp(max=100)
        soft=torch.zeros_like(logits)
        soft.scatter_add_(1,lower[:,None],(1-positions+lower)[:,None])
        soft.scatter_add_(1,upper[:,None],(positions-lower)[:,None])
        ce=-(soft*logits.log_softmax(-1)).sum(-1)
        loss=(ce*zero_target_weights(target,multiplier)*valid).sum()/valid.sum()
        loss.backward()
        return logits.grad,loss.detach()

    def test_zero_gradient_fivefold_nonzero_unchanged_padding_zero(self):
        base,_=self.gradients(1.)
        weighted,_=self.gradients(5.)
        torch.testing.assert_close(weighted[0],base[0]*5)
        torch.testing.assert_close(weighted[1:3],base[1:3])
        self.assertEqual(weighted[3].abs().sum().item(),0.)

    def test_all_zero_batch_not_normalized_by_sum_of_weights(self):
        gt=torch.zeros(4);valid=torch.ones(4,dtype=torch.bool)
        a,la=self.gradients(1.,gt,valid);b,lb=self.gradients(5.,gt,valid)
        torch.testing.assert_close(b,a*5);torch.testing.assert_close(lb,la*5)

    def test_small_positive_targets_not_boosted(self):
        torch.testing.assert_close(zero_target_weights(torch.tensor([0.,1e-9,.01,1.]),5.),torch.tensor([5.,1.,1.,1.]))

    def test_rank_partition_global_denominator(self):
        target=torch.tensor([0.,.3,0.,.7]);valid=torch.tensor([True,True,True,False])
        loss=torch.tensor([1.,2.,3.,4.],requires_grad=True)
        weights=zero_target_weights(target,5.)
        reference=(loss*weights*valid).sum()/valid.sum()
        # DDP averages local sums, then existing normalization multiplies world/n.
        distributed=sum((loss[i::2]*weights[i::2]*valid[i::2]).sum() for i in range(2))/2*(2/valid.sum())
        torch.testing.assert_close(distributed,reference)


if __name__=='__main__':unittest.main()
