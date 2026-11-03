import numpy as np
import pytest
from fastapi.testclient import TestClient

from mcaa.config import list_scenarios, load_scenario


@pytest.fixture(scope="module")
def client():
    from mcaa.api import app as appmod

    with TestClient(appmod.app) as c:
        yield c


def test_scenarios_validate():
    assert {"technician", "care_worker", "driver_sim"} <= set(list_scenarios())
    d = load_scenario("driver_sim")
    assert d.simulator_only and "SIMULATOR ONLY" in d.warning


def test_ask_and_adapt(client):
    r = client.post("/api/ask", json={"text": "What torque for the gland follower bolts?"}).json()
    assert r["plan"]["state"] == "IDLE" and "25 Nm" in r["plan"]["screen_text"]
    client.post("/api/context/override", json={"state": "HANDS_BUSY"})
    r = client.post("/api/ask", json={"text": "What torque for the gland follower bolts?", "source": "voice"}).json()
    assert r["plan"]["speak"] and r["plan"]["modality"] == "voice_first"
    client.post("/api/context/override", json={"state": None})


def test_plain_condition(client):
    client.post("/api/task/start", json={"task_set": "B", "condition": "B", "participant": "P02"})
    r = client.post("/api/ask", json={"text": "How often grease the bearings?"}).json()
    assert r["plan"]["state"] == "PLAIN" and not r["plan"]["speak"]
    client.post("/api/task/complete")
    assert client.delete("/api/study/participant/P02").json()["deleted_events"] > 0


def test_sensor_websocket_updates_context(client):
    from mcaa.sensors.synthetic import SubjectProfile, generate_segment

    seg = generate_segment("WALKING", 4, SubjectProfile(5))
    with client.websocket_connect("/ws/sensors") as ws:
        samples = [{"t": i / 50, **dict(zip(("ax", "ay", "az", "gx", "gy", "gz"), map(float, r), strict=False))} for i, r in enumerate(seg)]
        ws.send_json({"samples": samples})
        msg = ws.receive_json()
        assert msg["type"] == "ack" and msg["stats"]["received"] >= 128


def test_camera_consent_gate(client):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((60, 80, 3), np.uint8)).save(buf, "JPEG")
    body = {"jpeg_base64": base64.b64encode(buf.getvalue()).decode()}
    assert client.post("/api/camera/frame", json=body).status_code == 403
    client.post("/api/camera/consent", json={"consent": True})
    assert client.post("/api/camera/frame", json=body).status_code == 200
    client.post("/api/camera/consent", json={"consent": False})


def test_ui_served(client):
    assert "Context-Aware Assistant" in client.get("/").text
