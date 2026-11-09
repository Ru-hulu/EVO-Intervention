"""Scoped simulator entry; preserve native benchmark semantics, record three views."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
import capture

capture.install_base()
script = Path(sys.argv[1])
sys.argv = sys.argv[1:]
source = script.read_text()
replacements = {
    'task_env.cameras.get_third_view_rgb()': 'None',
    '            recorder.start()': '            recorder.start()\n            import capture\n            capture.start(task_env, output / "evidence" / str(now_id), now_seed, 20)',
    '    recorder.write(observation, third_person)': '    pass',
    '            recorder = FourViewRecorder(output, now_id, writer_factory)':
        '            class NullRecorder:\n                def start(self): pass\n                def close(self): capture.close()\n            recorder = NullRecorder()',
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise RuntimeError('Evaluator changed: review scoped video patch: '+old)
    source = source.replace(old, new)
exec(compile(source, str(script), 'exec'), {'__name__': '__main__', '__file__': str(script), '__package__': None})
