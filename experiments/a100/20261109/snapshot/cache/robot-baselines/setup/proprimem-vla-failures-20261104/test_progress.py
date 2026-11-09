import unittest,json
from publish import progress
class ProgressTests(unittest.TestCase):
 def test_overpress_is_not_completion(self):
  states=[dict(frame=i,poses={},success=False,press_cnt_1=n,press_cnt_2=0,press_cnt_check_button=0,card_id_1=1,card_id_2=1) for i,n in enumerate([0,1,2])]
  p,e=progress('press_button',states);self.assertEqual(p,[0,1/3,0]);self.assertEqual(e[-1]['kind'],'regression')
 def test_stall_keeps_state_instead_of_clock_ramp(self):
  states=[dict(frame=i,poses={},success=False,current_state_pointer=2,target_state_transition=list(range(7)),fail_flag=False) for i in range(100)]
  p,e=progress('cover_blocks',states);self.assertEqual(p,[1/3]*100);self.assertEqual(len(e),1)
 def test_terminal_failure_does_not_erase_valid_work(self):
  states=[dict(frame=i,poses={},success=False,current_state_pointer=k,target_state_transition=list(range(7)),fail_flag=False) for i,k in enumerate([0,1,2])]
  self.assertEqual(progress('cover_blocks',states)[0][-1],1/3)
 def test_native_irreversible_failure_has_evidence_drop(self):
  states=[dict(frame=i,poses={},success=False,current_state_pointer=2,target_state_transition=list(range(7)),fail_flag=bool(i)) for i in range(2)]
  p,e=progress('cover_blocks',states);self.assertEqual(p,[1/3,0]);self.assertTrue(e[-1]['evidence']['irreversible_failure'])
 def test_off_target_geometry_is_json_serializable(self):
  states=[dict(frame=0,poses={'block':[{'p':[.4,.3,.76]}]},success=False,stage_id=0,press_cnt=0)]
  dense,events=progress('put_back_block',states)
  self.assertEqual(dense,[0]);self.assertIs(events[0]['evidence']['checkpoint_seen'],False)
  json.dumps(events)
if __name__=='__main__':unittest.main()
