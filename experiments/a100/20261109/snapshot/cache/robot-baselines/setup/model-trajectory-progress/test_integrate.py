import unittest
from integrate import integrate
class IntegralTest(unittest.TestCase):
 def test_visual_motion_not_elapsed_time(self):
  b={'goals':['a','b'],'events':[{'frame':0,'done':[]},{'frame':3,'done':['a']},{'frame':5,'done':['a','b']}]}
  p,w,_=integrate([0,1,0,3,1,1],b,{'a':1,'b':3},True)
  self.assertEqual(p,[0,.0625,.0625,.25,.625,1.])
 def test_failure_does_not_reach_one(self):
  b={'goals':['a','b'],'events':[{'frame':0,'done':[]},{'frame':2,'done':['a']}]}
  p,_,_=integrate([0,1,1,500,500],b,{'a':1,'b':3},False)
  self.assertEqual(p,[0,.125,.25,.25,.25])
 def test_regression_restoration_no_double_credit(self):
  b={'goals':['a','b'],'events':[{'frame':0,'done':[]},{'frame':2,'done':['a']},{'frame':3,'done':[]},{'frame':5,'done':['a']},{'frame':6,'done':['a','b']}]}
  p,_,_=integrate([0,1,1,1,1,1,1],b,{'a':1,'b':1},True)
  self.assertEqual(p,[0,.25,.5,0,.25,.5,1])
 def test_zero_motion_completion_rejected(self):
  b={'goals':['a'],'events':[{'frame':0,'done':[]},{'frame':2,'done':['a']}]}
  with self.assertRaises(ValueError):integrate([0,0,0],b,{'a':1},True)
if __name__=='__main__':unittest.main()
