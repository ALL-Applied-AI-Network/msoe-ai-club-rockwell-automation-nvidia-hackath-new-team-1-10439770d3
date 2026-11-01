from robot_control.store import PlanStore


def test_plan_requires_human_approval(tmp_path):
    store = PlanStore(tmp_path / "plans.sqlite3")
    plan = store.create("test", [{"x_mm": 450, "y_mm": 0, "z_mm": -230}])
    assert plan["status"] == "pending"
    assert store.claim_next("mock") is None
    approved = store.approve(plan["id"], "mock", "Test Operator")
    assert approved["status"] == "approved"
    claimed = store.claim_next("mock")
    assert claimed["id"] == plan["id"]
    assert claimed["status"] == "executing"
    assert store.complete(plan["id"])["status"] == "completed"


def test_executing_plan_can_be_closed_as_interrupted(tmp_path):
    store = PlanStore(tmp_path / "plans.sqlite3")
    plan = store.create("test", [{"x_mm": 450, "y_mm": 0, "z_mm": -230}])
    store.approve(plan["id"], "physical", "Test Operator")
    store.claim_next("physical")

    interrupted = store.interrupt(plan["id"], "operator ended rollout")

    assert interrupted["status"] == "interrupted"
    assert interrupted["note"] == "operator ended rollout"
