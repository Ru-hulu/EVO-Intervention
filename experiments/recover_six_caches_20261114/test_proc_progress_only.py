from decimal import Decimal
import unittest

from .proc_progress_only import answer_text,supervised_ids


class ProgressOnlyTests(unittest.TestCase):
    def test_preserves_progress_without_reasoning(self):
        for pt in [0.,1.,0.12345678901234567,0.7,1e-12]:
            text=answer_text(pt)
            self.assertEqual(Decimal(text[10:-12])/100,Decimal(str(pt)))
            self.assertNotIn('think',text)
        self.assertEqual(answer_text(1),'<progress>100%</progress>')
        self.assertEqual(answer_text(0),'<progress>0%</progress>')

    def test_rejects_bad_targets(self):
        for pt in [float('nan'),float('inf'),-0.1,1.1,True]:
            with self.assertRaises(ValueError):answer_text(pt)

    def test_only_answer_receives_loss(self):
        ids,labels=supervised_ids([1,2,3],[4,5,6],6)
        self.assertEqual(ids,[1,2,3,4,5,6])
        self.assertEqual(labels,[-100,-100,-100,4,5,6])
        with self.assertRaises(ValueError):supervised_ids([1,2],[3,4],3)


if __name__=='__main__':unittest.main()
