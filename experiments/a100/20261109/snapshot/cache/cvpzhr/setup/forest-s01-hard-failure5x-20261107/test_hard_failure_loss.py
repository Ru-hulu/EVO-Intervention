import unittest
import torch
from hard_failure_loss import hard_failure_weights, FailureCollator


class Tests(unittest.TestCase):
    def test_only_failed_zero_and_high_predictions_boosted(self):
        # Initial zero, positive target, recovered/normal, easy failure, hard failure, exact threshold.
        target=torch.tensor([0.,.7,0.,0.,0.,0.])
        pred=torch.tensor([.9,.9,.9,.01,.9,.2],requires_grad=True)
        native=torch.tensor([False,True,False,True,True,True])
        weights,mask=hard_failure_weights(target,pred,native)
        torch.testing.assert_close(weights,torch.tensor([1.,1.,1.,1.,5.,1.]))
        self.assertFalse(weights.requires_grad)
        self.assertEqual(mask.tolist(),[False,False,False,False,True,False])

    def test_gradients_and_padding(self):
        target=torch.tensor([0.,0.,.5,0.]);native=torch.tensor([False,True,True,True]);valid=torch.tensor([1.,1.,1.,0.])
        logits=torch.zeros(4,101,requires_grad=True)
        probs=logits.softmax(-1);pred=probs@torch.linspace(0,1,101)
        targets=torch.zeros_like(logits);targets[0,0]=targets[1,0]=targets[3,0]=1;targets[2,50]=1
        ce=-(targets*logits.log_softmax(-1)).sum(-1)
        weights,_=hard_failure_weights(target,pred,native)
        base=torch.autograd.grad((ce*valid).sum()/3,logits,retain_graph=True)[0]
        grad=torch.autograd.grad((ce*weights*valid).sum()/3,logits)[0]
        torch.testing.assert_close(grad[0],base[0]);torch.testing.assert_close(grad[1],base[1]*5)
        torch.testing.assert_close(grad[2],base[2]);self.assertEqual(grad[3].abs().sum().item(),0)

    def test_failure_metadata_never_enters_model_inputs(self):
        inputs={'unchanged':torch.ones(1)}
        collator=FailureCollator(lambda samples:dict(inputs=inputs,target_pt=torch.zeros(1)),{'ep':{'13':True}})
        result=collator([dict(episode_id='ep',frame_index=13)])
        self.assertIs(result['inputs'],inputs)
        self.assertEqual(list(inputs),['unchanged'])
        self.assertEqual(result['native_failure'].tolist(),[True])


if __name__=='__main__':unittest.main()
