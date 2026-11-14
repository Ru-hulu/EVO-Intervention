from . import controller


def test_adopted_worker_returncode_after_exit(monkeypatch):
    monkeypatch.setattr(controller, 'alive', lambda pid: True)
    process = controller.Adopted(123, 'test')
    assert process.poll() is None and process.returncode is None
    monkeypatch.setattr(controller, 'alive', lambda pid: False)
    monkeypatch.setattr(controller, 'read', lambda path: {'status': 'completed'})
    assert process.poll() == process.returncode == 0


def test_adopted_failure_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(controller, 'alive', lambda pid: False)
    monkeypatch.setattr(controller, 'read', lambda path: {'status': 'failed'})
    process = controller.Adopted(123, 'test')
    assert process.poll() == process.returncode == 1
