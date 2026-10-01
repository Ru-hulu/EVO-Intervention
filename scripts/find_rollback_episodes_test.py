from scripts.find_rollback_episodes import rollback_events


def stage(nowid, restid, frame):
    return {"nowid": nowid, "restid": restid, "name": nowid, "30FPSstart_frame": frame}


def test_initial_tasks_and_normal_order_do_not_count_as_rollback():
    assert (
        rollback_events([stage("2", "1,3,4", 0), stage("1", "3,4", 10), stage("3", "4", 20), stage("4", "", 30)]) == []
    )


def test_reset_can_happen_before_the_failed_task_is_repeated():
    events = rollback_events([stage("2", "1,3,4", 0), stage("1", "2,3,4", 10), stage("2", "3,4", 20)])
    assert [(event["frame"], event["reset_ids"]) for event in events] == [(10, ["2"])]


def test_retry_resets_completed_dependencies_but_not_unfinished_ids():
    events = rollback_events(
        [
            stage("1", "2,3,4", 0),
            stage("2", "3,4", 10),
            stage("3", "4", 20),
            stage("2", "3,4", 30),
            stage("3", "4", 40),
            stage("4", "", 50),
        ]
    )
    assert [(event["frame"], event["reset_ids"]) for event in events] == [(30, ["2", "3"])]
