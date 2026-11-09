import unittest
import torch
from progress_loss import progress_objective


class Tests(unittest.TestCase):
    def test_symmetric_and_large_errors(self):
        pred=torch.tensor([.1,.9,0.,.9]);gt=torch.tensor([0.,0.,.9,.9])
        loss=progress_objective(torch.ones(4),pred,gt,5.)
        torch.testing.assert_close(loss,torch.tensor([1.05,5.05,5.05,1.]))

    def test_gradient_reaches_distribution_and_padding_is_masked(self):
        logits=torch.zeros(3,101,requires_grad=True)
        probs=logits.softmax(-1);pred=(probs*torch.linspace(0,1,101)).sum(-1)
        gt=torch.tensor([0.,1.,0.]);ce=-logits.log_softmax(-1)[torch.arange(3),torch.tensor([0,100,0])]
        mask=torch.tensor([1.,1.,0.]);loss=(progress_objective(ce,pred,gt,5.)*mask).sum()/mask.sum()
        base_grad=torch.autograd.grad((ce*mask).sum()/mask.sum(),logits,retain_graph=True)[0]
        grad=torch.autograd.grad(loss,logits)[0]
        self.assertGreater(grad[0,100].item(),base_grad[0,100].item())
        self.assertGreater(grad[1,0].item(),base_grad[1,0].item())
        self.assertEqual(grad[2].abs().sum().item(),0.)
        self.assertTrue(torch.isfinite(grad).all())

    def test_zero_coefficient_preserves_ce(self):
        ce=torch.tensor([.3,.7]);torch.testing.assert_close(progress_objective(ce,torch.ones(2),torch.zeros(2),0.),ce)


if __name__=='__main__':unittest.main()
