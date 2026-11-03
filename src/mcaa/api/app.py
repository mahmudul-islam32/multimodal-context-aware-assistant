"""FastAPI backend (F-10 server side): REST, WebSockets and the PWA.

    uvicorn mcaa.api.app:app --reload        # or: mcaa-api
    open http://localhost:8000               # tablet UI
    open http://localhost:8000/sensor.html   # on the phone: stream IMU
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..assistant import Assistant
from ..config import list_scenarios
from ..sensors.synthetic import ACTIVITIES, SubjectProfile, generate_segment
from ..speech.asr import decode_wav, get_asr
from ..study.conditions import schedule_for, validate_pid
from ..vision import ConsentRequired

STATIC = Path(__file__).resolve().parents[1] / "web" / "static"


class Hub:
    """Fan-out of state updates to connected UIs."""

    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def broadcast(self, msg: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


class State:
    assistant: Assistant
    hub = Hub()
    sim_task: asyncio.Task | None = None
    sim_activity: str | None = None
    asr = None


S = State()


def build(scenario: str | None = None) -> Assistant:
    a = Assistant(scenario or os.environ.get("MCAA_SCENARIO", "technician"))
    a.ensure_har()
    return a


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.assistant = build()
    yield
    if S.sim_task:
        S.sim_task.cancel()


app = FastAPI(title="Multimodal Context-Aware Assistant", version="0.1.0", lifespan=lifespan)


# ------------------------------------------------------------------ models
class AskIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    source: str = "text"
    asr_confidence: float | None = None
    noise_dbfs: float | None = None


class OverrideIn(BaseModel):
    state: str | None = None
    duration_s: float | None = None


class StartIn(BaseModel):
    task_set: str = "A"
    condition: str = "A"
    participant: str | None = None


class StepIn(BaseModel):
    index: int


class ErrorIn(BaseModel):
    kind: str
    note: str = ""


class SimIn(BaseModel):
    activity: str | None = None


class AudioLevelIn(BaseModel):
    noise_dbfs: float


class ConsentIn(BaseModel):
    consent: bool


class FrameIn(BaseModel):
    jpeg_base64: str


async def push_state(extra: dict | None = None):
    await S.hub.broadcast({"type": "state", **S.assistant.snapshot(), **(extra or {})})


# ------------------------------------------------------------------ REST
@app.get("/api/health")
def health():
    return {"ok": True, "scenario": S.assistant.scenario.id, "generator": S.assistant.rag.generator.name}


@app.get("/api/scenarios")
def scenarios():
    return {"available": list_scenarios(), "active": S.assistant.scenario.id}


@app.post("/api/scenarios/{name}")
async def switch_scenario(name: str):
    if name not in list_scenarios():
        raise HTTPException(404, "unknown scenario")
    S.assistant = build(name)
    await push_state()
    return S.assistant.snapshot()


@app.get("/api/state")
def state():
    return S.assistant.snapshot()


@app.post("/api/ask")
async def ask(body: AskIn):
    res = await asyncio.to_thread(S.assistant.ask, body.text, body.source, body.asr_confidence, body.noise_dbfs)
    await push_state()
    return res


@app.post("/api/touch")
async def touch():
    S.assistant.touch()
    return {"ok": True}


@app.post("/api/context/override")
async def override(body: OverrideIn):
    res = S.assistant.override(body.state, body.duration_s)
    await push_state()
    return res


@app.post("/api/audio/level")
async def audio_level(body: AudioLevelIn):
    st = S.assistant.context.update_audio(noise_dbfs=body.noise_dbfs)
    await push_state()
    return st.to_dict()


@app.post("/api/task/start")
async def task_start(body: StartIn):
    if body.participant:
        validate_pid(body.participant)
    res = S.assistant.start_task(body.task_set, body.condition, body.participant)
    await push_state()
    return res


@app.post("/api/task/step")
async def task_step(body: StepIn):
    res = S.assistant.set_step(body.index)
    await push_state()
    return res


@app.post("/api/task/complete")
async def task_complete():
    res = S.assistant.complete_task()
    await push_state()
    return res


@app.post("/api/task/error")
def task_error(body: ErrorIn):
    S.assistant.record_error(body.kind, body.note)
    return {"ok": True}


@app.post("/api/speech/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Push-to-talk upload (16-bit WAV). Audio is transcribed in memory and discarded."""
    if S.asr is None:
        S.asr = get_asr()
    audio, sr = decode_wav(await file.read())
    try:
        r = await asyncio.to_thread(S.asr.transcribe, audio, sr)
    except RuntimeError as e:
        raise HTTPException(501, str(e)) from e
    return {"text": r.text, "confidence": r.confidence, "latency_ms": r.latency_ms, "noise_dbfs": r.noise_dbfs, "backend": r.backend}


@app.get("/api/study/schedule/{pid}")
def schedule(pid: str):
    try:
        return {"participant": pid, "blocks": schedule_for(pid)}
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/study/export")
def export(fmt: str = "csv"):
    out = Path(os.environ.get("MCAA_EXPORT_DIR", "results/study/export"))
    paths = S.assistant.logger.export(out, fmt)
    return {"files": [str(p) for p in paths]}


@app.get("/api/log")
def events(type: str | None = None, limit: int = 50):
    df = S.assistant.logger.events(type)
    return df.tail(limit).to_dict(orient="records")


@app.delete("/api/study/participant/{pid}")
def delete_participant(pid: str):
    validate_pid(pid)
    return {"deleted_events": S.assistant.logger.delete_participant(pid)}


@app.post("/api/camera/consent")
async def camera_consent(body: ConsentIn):
    if body.consent:
        S.assistant.camera.enable(True)
    else:
        S.assistant.camera.disable()
    S.assistant.logger.log("camera_consent", consent=body.consent)
    await push_state()
    return {"enabled": S.assistant.camera.enabled}


@app.post("/api/camera/frame")
async def camera_frame(body: FrameIn):
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(base64.b64decode(body.jpeg_base64.split(",")[-1]))).convert("RGB")
        img.thumbnail((160, 120))
        res = S.assistant.camera_frame(np.asarray(img))
    except ConsentRequired as e:
        raise HTTPException(403, str(e)) from e
    return res


# ------------------------------------------------------------------ simulated sensor
async def _sim_loop():
    """Stream synthetic IMU at 50 Hz in real time (demo without a phone)."""
    prof = SubjectProfile(77, seed=3)
    t0 = time.time()
    n = 0
    while True:
        act = S.sim_activity
        if act is None:
            await asyncio.sleep(0.1)
            continue
        block = generate_segment(act, 0.5, prof)
        for row in block:
            sample = {"t": t0 + n / 50.0, **dict(zip(("ax", "ay", "az", "gx", "gy", "gz"), map(float, row), strict=True))}
            n += 1
            upd = S.assistant.sensor_sample(sample)
            if upd:
                await push_state({"har_ms": upd["har_ms"]})
        await asyncio.sleep(0.5)


@app.post("/api/sim/activity")
async def sim_activity(body: SimIn):
    if body.activity and body.activity not in ACTIVITIES:
        raise HTTPException(422, f"activity must be one of {ACTIVITIES}")
    S.sim_activity = body.activity
    if body.activity and (S.sim_task is None or S.sim_task.done()):
        S.sim_task = asyncio.create_task(_sim_loop())
    return {"activity": S.sim_activity}


# ------------------------------------------------------------------ websockets
@app.websocket("/ws/ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    S.hub.clients.add(ws)
    await ws.send_json({"type": "state", **S.assistant.snapshot()})
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "touch":
                S.assistant.touch()
    except WebSocketDisconnect:
        S.hub.clients.discard(ws)


@app.websocket("/ws/sensors")
async def ws_sensors(ws: WebSocket):
    """Phone streams batches: {"samples": [{"t", "ax".."gz", "sent_at"}...]}.
    Accelerometer in g (or m/s^2 with "unit": "ms2"), gyro in rad/s."""
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            scale = 1 / 9.81 if msg.get("unit") == "ms2" else 1.0
            for s in msg.get("samples", []):
                for k in ("ax", "ay", "az"):
                    s[k] = float(s.get(k, 0)) * scale
                upd = S.assistant.sensor_sample(s)
                if upd:
                    await push_state({"har_ms": upd["har_ms"]})
                    await ws.send_json({"type": "ack", "context": upd["context"]["state"], "stats": S.assistant.sensors.stats.summary()})
    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------------ static PWA
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn

    uvicorn.run("mcaa.api.app:app", host=os.environ.get("MCAA_HOST", "0.0.0.0"), port=int(os.environ.get("MCAA_PORT", "8000")))


if __name__ == "__main__":
    main()
