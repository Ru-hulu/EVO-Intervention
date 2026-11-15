import unittest
from .prepare import choose


class Selection(unittest.TestCase):
    def test_fixed_selection_excludes_training(self):
        task={'train':list(range(1,26)),'val':list(range(29,36))}
        self.assertEqual(choose(task,list(range(1,21))),34)
        self.assertEqual(choose(task,list(range(1,21))),choose(task,list(range(1,21))))

    def test_overlap_rejected(self):
        with self.assertRaises(AssertionError):choose({'train':[1,2],'val':[2,3]},[1])
        with self.assertRaises(AssertionError):choose({'train':[1],'val':[2,3]},[2])


if __name__=='__main__':unittest.main()
