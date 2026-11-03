import pytest

from mcaa.config import load_context_config, load_policy_config
from mcaa.context import ContextEngine
from mcaa.policy import OutputPolicy
from mcaa.policy.policy import truncate_words


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def eng():
    c = Clock()
    e = ContextEngine(load_context_config(), clock=c)
    e.touch()
    return e, c


def test_hysteresis_requires_confirmation_and_dwell(eng):
    e, c = eng
    # EMA (alpha 0.5): p_moving 0.5 -> 0.75 -> 0.875; MOVING proposed from the 2nd update
    for t, expected in [(10, "IDLE"), (11, "IDLE"), (12, "MOVING")]:
        c.t = t
        assert e.update_activity({"WALKING": 1.0}).state == expected
    # back to standing: confirmed after 2 proposals but dwell (4 s) blocks the change
    for t in (13, 14, 15):
        c.t = t
        assert e.update_activity({"STANDING": 1.0}).state == "MOVING"
    c.t = 16.5
    assert e.update_activity({"STANDING": 1.0}).state == "IDLE"
    assert [h.state for h in e.history] == ["IDLE", "MOVING", "IDLE"]


def test_manual_step_without_touch_becomes_hands_busy(eng):
    e, c = eng
    e.set_step(True)
    c.t = 30
    e.evaluate()
    c.t = 31
    assert e.evaluate().state == "HANDS_BUSY"


def test_noise_rule(eng):
    e, c = eng
    c.t = 10
    e.update_audio(noise_dbfs=-10)
    c.t = 11
    assert e.update_audio(noise_dbfs=-10).state == "NOISY"


def test_override_wins_and_expires(eng):
    e, c = eng
    e.override("NOISY", duration_s=5)
    assert e.current.state == "NOISY" and e.current.overridden
    c.t = 10
    e.evaluate()
    c.t = 11
    assert e.evaluate().state == "IDLE"


def test_policy_limits():
    pol = OutputPolicy(load_policy_config())
    long = "Tighten the gland follower bolts to 25 Nm in a cross pattern. [1] " * 5
    cites = [{"title": "Pump manual", "section": "section 4.3"}]
    busy = pol.plan("HANDS_BUSY", long, "Tighten the bolts to 25 Nm. [1]", cites, False)
    assert busy.speak and busy.modality == "voice_first" and len(busy.voice_text.split()) <= 25 + 8
    assert "[1]" not in busy.voice_text and "Pump manual" in busy.voice_text
    idle = pol.plan("IDLE", long, "short [1]", cites, False)
    assert idle.screen_text == long and idle.modality == "screen"
    noisy = pol.plan("NOISY", long, "short [1]", cites, False)
    assert not noisy.speak and noisy.visual_cue
    plain = pol.plan("HANDS_BUSY", long, "short", cites, False, plain=True)
    assert plain.state == "PLAIN" and not plain.speak


def test_truncate_words():
    assert truncate_words("One two three. Four five six seven.", 4) == "One two three."
    assert truncate_words("a b c d e f", 3) == "a b c…"
