"""The memory guard (preflight refusals) and the watchdog (unload while loaded), with faked machine readings."""
import json
import os
import sys
import threading
import time

import pytest

from conftest import FAKE_ENGINE, FakeProbes, reply
from inkling_shim.engine import Config, EngineBlocked, Manager, MemoryTrip
from inkling_shim.memguard import GuardConfig, MemoryGuard, SystemProbes, engine_need_gb
from inkling_shim.service import Service
from test_server import Harness, MSG, wait_for
from inkling_shim.server import make_server


# Every optional check switched on (the probes are fake, so the paths are never touched); 64 GiB of RAM.
FULL = dict(lock_dir="/nonexistent/lock", health_url="http://127.0.0.1:9/health", pause_file="/nonexistent/paused",
            watch_file="/nonexistent/service.log", total_gb=64.0)


def guard(**probes):
    return MemoryGuard(18.7, GuardConfig(poll_s=0.05, cooldown_s=0.0, **FULL), FakeProbes(**probes))


def test_need_gb_by_context():
    assert engine_need_gb(32768) == 17.7          # 16.46 GiB measured
    assert engine_need_gb(65536) == 18.7
    assert engine_need_gb(131072) == 20.9
    assert engine_need_gb(32768, "auto") == 18.1  # 16.90 GiB measured with DSpark


def test_preflight_all_clear():
    assert guard().preflight() == []


@pytest.mark.parametrize("probes, words", [
    ({"holder": "bench (pid 1, since 13:00)"}, "guarded GPU job"),
    ({"healthy": False}, "not healthy and not marked paused"),
    ({"level": 2}, "memory pressure level 2"),
    ({"level": -1}, "memory pressure level -1"),
    ({"free": 40}, "only 25.6 GB free"),          # 18.7 + 8 needed
    ({"free": -1}, "free"),
    ({"procs": {"llama-server": {11}}}, "llama-server pid [11]"),
    ({"procs": {"strata-inkling": {12}}}, "strata-inkling pid [12]"),
    ({"ollama": ["arete"]}, "Ollama model arete"),
])
def test_preflight_refusals(probes, words):
    problems = guard(**probes).preflight()
    assert len(problems) == 1 and words in problems[0]


def test_optional_checks_are_off_by_default():
    """Without lock_dir / health_url / pause_file / watch_file only the memory checks run."""
    g = MemoryGuard(18.7, GuardConfig(poll_s=0.05, cooldown_s=0.0, total_gb=64.0),
                    FakeProbes(holder="job (pid 1)", healthy=False, procs={"llama-server": {11}}, ollama=["x"]))
    assert g.preflight() == []
    g.arm()
    g.probes.ooms = 99
    assert g.check() is None
    assert "free" in MemoryGuard(18.7, GuardConfig(total_gb=64.0), FakeProbes(free=30)).preflight()[0]


def test_paused_service_allows_sharing_but_not_low_memory():
    assert guard(paused=True, healthy=False, procs={"llama-server": {11}}, ollama=["x"]).preflight() == []
    assert "free" in guard(paused=True, free=30).preflight()[0]
    assert "pressure" in guard(paused=True, level=4).preflight()[0]


def test_watchdog_checks():
    g = guard()
    g.arm()
    assert g.check() is None
    g.probes.ooms = 8
    assert "insufficient memory" in g.check()
    g = guard()
    g.arm()
    g.probes.level = 4                                # critical: at once
    assert "critical" in g.check()
    g.probes.level, g.probes.free = 1, 11
    assert "11% < floor 12%" in g.check()
    g.probes.free, g.probes.ooms = 12, -1             # unreadable log: no false alarm
    assert g.check() is None


def test_watchdog_tolerates_a_brief_pressure_warning():
    g = MemoryGuard(18.7, GuardConfig(poll_s=0.05, cooldown_s=0.0, warn_grace_s=0.3, **FULL), FakeProbes())
    g.arm()
    g.probes.level = 2
    assert g.check() is None                          # a warning starts the clock, nothing more
    time.sleep(0.1)
    assert g.check() is None
    g.probes.level = 1                                # it cleared: the clock resets
    assert g.check() is None
    g.probes.level = 2
    assert g.check() is None
    time.sleep(0.35)
    assert "pressure level 2 for" in g.check()        # a warning that does not clear unloads the engine


def test_system_probes_read_this_machine():
    """Read-only probes on the real Mac (ps, sysctl, memory_pressure): they parse, whatever the values are."""
    p = SystemProbes(GuardConfig())
    assert p.pressure_level() in (1, 2, 4)
    assert 0 <= p.free_pct() <= 100
    assert isinstance(p.processes("launchd"), set) and p.processes("launchd")
    assert p.paused() is False and p.healthy() is True and p.watch_count() == -1   # nothing configured
    assert p.cfg.total_gb > 0


def test_lock_holder(tmp_path):
    p = SystemProbes(GuardConfig(lock_dir=str(tmp_path / "lock")))
    assert p.lock_holder() is None
    (tmp_path / "lock").mkdir()
    (tmp_path / "lock" / "owner.json").write_text(json.dumps({"pid": os.getpid(), "name": "job", "since": "x"}))
    assert "job (pid" in p.lock_holder()
    (tmp_path / "lock" / "owner.json").write_text(json.dumps({"pid": 99999999, "name": "gone"}))
    assert p.lock_holder() is None                   # stale lock: owner is gone


@pytest.fixture
def served(tmp_path, renderer, monkeypatch):
    made = []

    def _make(replies, g, **script):
        sp = tmp_path / f"s{len(made)}.json"
        script.setdefault("gate", str(tmp_path / f"gate{len(made)}"))
        open(script["gate"], "w").close()
        sp.write_text(json.dumps({"replies": replies, **script}))
        log_path = tmp_path / f"fake{len(made)}.log"
        monkeypatch.setenv("FAKE_SCRIPT", str(sp))
        monkeypatch.setenv("FAKE_LOG", str(log_path))
        cfg = Config(engine=[sys.executable, str(FAKE_ENGINE)], pack="fake.strata", ctx=4096, idle=0,
                     log_path=str(tmp_path / "engine.log"), silence_timeout=20)
        logs = []
        m = Manager(cfg, guard=g, log=logs.append)
        server = make_server(Service(renderer, m, log=logs.append), "test-model", 0)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        h = Harness(server, m, None, log_path, logs)
        h.gate = script["gate"]
        made.append(h)
        return h

    yield _make
    for h in made:
        if os.path.exists(h.gate):
            os.unlink(h.gate)
        h.server.shutdown()
        h.manager.shutdown(timeout=10)
        h.server.server_close()


def test_watchdog_unloads_mid_request(served, vocab):
    g = guard()
    g.cfg.cooldown_s = 60
    h = served([reply(vocab, text="never")], g)
    out = []
    t = threading.Thread(target=lambda: out.append(h.post({"messages": MSG}, 60)))
    t.start()
    assert wait_for(lambda: h.logged(cmd="GENERATE"))  # held mid-request by the gate
    proc = h.manager.engine.proc
    g.probes.ooms += 1                                  # the watched log gains "insufficient memory"
    t.join(60)
    st, body = out[0]
    assert st == 503 and body["error"]["code"] == "memory_watchdog" and "insufficient memory" in body["error"]["message"]
    assert proc.poll() is not None and h.manager.engine is None
    assert wait_for(lambda: any("MEMORY WATCHDOG" in x for x in h.logs))
    st, body = h.post({"messages": MSG})                # cooldown: refuses to load again straight away
    assert st == 503 and "not loading again" in body["error"]["message"]


def test_watchdog_unloads_idle_engine_then_reloads_after_cooldown(served, vocab):
    g = guard()
    h = served([reply(vocab, text="a")], g)
    os.unlink(h.gate)
    assert h.post({"messages": MSG})[0] == 200
    proc = h.manager.engine.proc
    g.probes.free = 5
    assert wait_for(lambda: proc.poll() is not None and h.manager.engine is None)
    st, body = h.post({"messages": MSG})                # still low memory: preflight refuses
    assert st == 503 and "GB free" in body["error"]["message"]
    g.probes.free = 80
    assert h.post({"messages": MSG})[0] == 200 and h.manager.loads == 2


def test_watchdog_during_load(served, vocab, tmp_path):
    g = guard()
    h = served([reply(vocab, text="a")], g, load_delay=3)
    out = []
    t = threading.Thread(target=lambda: out.append(h.post({"messages": MSG}, 60)))
    t.start()
    assert wait_for(lambda: h.manager.state == "loading" and h.manager._loading is not None)
    g.probes.level = 4
    t.join(60)
    assert out[0][0] == 503 and out[0][1]["error"]["code"] == "memory_watchdog"
    assert h.manager.state == "unloaded" and h.manager.engine is None


def test_lock_appearing_while_loaded_does_not_unload(served, vocab):
    """A guarded job taking the external lock blocks new loads, never a loaded engine."""
    g = guard()
    h = served([reply(vocab, text="a")], g)
    os.unlink(h.gate)
    assert h.post({"messages": MSG})[0] == 200
    proc = h.manager.engine.proc
    g.probes.holder = "calibration (pid 1, since 16:00)"
    time.sleep(0.5)                                    # ten watchdog polls
    assert proc.poll() is None and h.manager.engine is not None
    assert h.post({"messages": MSG})[0] == 200        # the loaded engine keeps serving
    h.manager.unload()
    st, body = h.post({"messages": MSG})
    assert st == 503 and "guarded GPU job" in body["error"]["message"]
