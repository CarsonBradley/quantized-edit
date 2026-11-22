"""Memory safety for a long-lived engine on a unified-memory Mac.

The engine allocates its whole KV cache when it loads (about 17.5 GiB at 64K context), and macOS does not refuse
a GPU allocation that does not fit: it compresses or swaps other processes' memory instead. So the shim checks
before it loads the engine, and watches while it is loaded.

Always on (unless the server runs with --no-guard):
  before loading (`preflight`, a refusal is answered with HTTP 503):
    * memory pressure is normal (kern.memorystatus_vm_pressure_level == 1);
    * free or reclaimable memory >= need_gb (the engine's size at the chosen context) + reserve_gb;
  while loaded (`check`, every poll_s seconds; a hit unloads the engine):
    * memory pressure goes critical (level 4, at once), or stays at warning (level 2) for warn_grace_s
      (default 0: the next poll);
    * free or reclaimable memory drops under floor_pct (at once).

Optional, all off by default (GuardConfig fields; INKLING_S_* settings in server.py):
  lock_dir      an external lock directory (with owner.json {"pid": N, "name": ..., "since": ...}) that another
                tool holds while it runs a GPU job: no load while it is held by a live process. A loaded engine
                is not unloaded when the lock appears.
  health_url    a service that shares the machine and must stay healthy (HTTP 200): no load while it is not.
                While it is up, no other large model may be loaded at the same time (llama-server, an Ollama
                model, another strata-inkling).
  pause_file    if this file exists, the service behind health_url counts as paused: its health is not
                required, and other models may share the machine when the memory check passes.
  watch_file    a log to watch (for example the other service's log), with watch_pattern (default
                "insufficient memory", the Metal out-of-memory message): a new occurrence unloads the engine
                at once.
Every probe is a method of `SystemProbes`, so tests swap in a fake.
"""
from __future__ import annotations

import json
import os
import time
import subprocess
import urllib.request
from dataclasses import dataclass


def _sh(cmd, timeout=20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def physical_gib() -> float:
    """Installed RAM in GiB (sysctl hw.memsize; sysconf as a fallback)."""
    try:
        return int(_sh(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip()) / 2**30
    except ValueError:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except (ValueError, OSError):
            return 0.0


@dataclass
class GuardConfig:
    lock_dir: str | None = None
    health_url: str | None = None
    pause_file: str | None = None
    watch_file: str | None = None
    watch_pattern: str = "insufficient memory"
    total_gb: float | None = None      # None: the installed RAM
    reserve_gb: float = 8.0
    floor_pct: int = 12
    # Pressure level 2 (warning) is tolerated this long before unloading. 0 = unload on the next poll.
    # A 20 s grace was tried beside another large GPU workload: the warning did not clear, and about 21 GB of
    # the other workload's memory went to the compressor while the engine stayed up.
    warn_grace_s: float = 0.0
    poll_s: float = 2.0
    cooldown_s: float = 120.0          # after the watchdog unloaded the engine, refuse to load it for this long

    def __post_init__(self):
        if self.total_gb is None:
            self.total_gb = physical_gib() or 64.0


def engine_need_gb(ctx: int, draft: str = "off") -> float:
    """The engine's memory at a context size, in GB (1e9), from the measured TotalAlloc (16.46 GiB at 32K with
    drafting off, 16.90 with it on) and the source (fp16 KV: 1 GiB per 32K; DSpark KV: 0.375 GiB per 32K + fixed)."""
    gib = 15.46 + ctx / 32768 + (0.44 * ctx / 32768 if draft != "off" else 0.0)
    return round(gib * 1.073741824, 1)


class SystemProbes:
    """The real machine."""

    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg

    def paused(self) -> bool:
        return bool(self.cfg.pause_file) and os.path.exists(self.cfg.pause_file)

    def healthy(self) -> bool:
        if not self.cfg.health_url:
            return True
        try:
            return urllib.request.urlopen(self.cfg.health_url, timeout=4).status == 200
        except Exception:
            return False

    def watch_count(self) -> int:
        """Occurrences of watch_pattern in watch_file (case-insensitive); -1 if unset or unreadable."""
        if not self.cfg.watch_file:
            return -1
        try:
            with open(self.cfg.watch_file, "rb") as f:
                return f.read().lower().count(self.cfg.watch_pattern.lower().encode())
        except OSError:
            return -1

    def free_pct(self) -> int:
        """Percent of RAM free or reclaimable without swapping: free + speculative + purgeable + file-backed
        (page cache) pages. (`memory_pressure` counts page cache as used, so reading the 13 GB pack looked
        like exhaustion.)"""
        page, v = 16384, {}
        for line in _sh(["/usr/bin/vm_stat"]).splitlines():
            if "page size of" in line:
                try:
                    page = int(line.split("page size of")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            elif ":" in line:
                k, _, val = line.partition(":")
                try:
                    v[k.strip().strip('"')] = int(val.strip().rstrip("."))
                except ValueError:
                    pass
        try:
            pages = v["Pages free"] + v["Pages speculative"] + v["Pages purgeable"] + v["File-backed pages"]
        except KeyError:
            return -1
        return int(pages * page / (self.cfg.total_gb * 2**30) * 100)

    def pressure_level(self) -> int:
        try:
            return int(_sh(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip())
        except ValueError:
            return -1

    def processes(self, prefix: str) -> set[int]:
        """pids whose executable name starts with prefix (not command lines that merely mention it)."""
        found = set()
        for line in _sh(["/bin/ps", "-axo", "pid=,comm="]).splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and os.path.basename(parts[1].strip()).startswith(prefix):
                found.add(int(parts[0]))
        return found

    def ollama_models(self) -> list[str]:
        out = _sh(["/bin/sh", "-c", "ollama ps 2>/dev/null | tail -n +2"])
        return [line.split()[0] for line in out.splitlines() if line.strip()]

    def lock_holder(self) -> str | None:
        """None if no lock_dir, or no job holds it; a description of the holder otherwise (a lock whose owner is
        gone: None)."""
        lock = self.cfg.lock_dir
        if not lock or not os.path.isdir(lock):
            return None
        try:
            with open(os.path.join(lock, "owner.json")) as f:
                info = json.load(f)
            os.kill(int(info["pid"]), 0)
            return f"{info.get('name')} (pid {info['pid']}, since {info.get('since')})"
        except ProcessLookupError:
            return None                  # stale lock: its owner is gone
        except (OSError, ValueError, KeyError):
            return "unknown owner"


class MemoryGuard:
    def __init__(self, need_gb: float, cfg: GuardConfig | None = None, probes=None):
        self.cfg = cfg or GuardConfig()
        self.probes = probes or SystemProbes(self.cfg)
        self.need_gb = need_gb
        self.watch_base = -1
        self.warn_since: float | None = None   # when pressure level 2 was first seen in the current spell

    @property
    def poll_s(self) -> float:
        return self.cfg.poll_s

    def describe(self) -> str:
        c = self.cfg
        opts = [f"lock {c.lock_dir}" if c.lock_dir else None, f"health {c.health_url}" if c.health_url else None,
                f"pause file {c.pause_file}" if c.pause_file else None,
                f"watch {c.watch_file} for '{c.watch_pattern}'" if c.watch_file else None]
        return (f"needs {self.need_gb:.1f} GB + {c.reserve_gb:.0f} GB reserve of {c.total_gb:.0f} GiB; "
                + (", ".join(o for o in opts if o) or "no optional checks"))

    def preflight(self, own_pids=()) -> list[str]:
        p, c = self.probes, self.cfg
        problems = []
        holder = p.lock_holder() if c.lock_dir else None
        if holder:
            problems.append(f"a guarded GPU job is running ({holder}; lock {c.lock_dir})")
        paused = bool(c.pause_file) and p.paused()
        protected = bool(c.health_url) and not paused
        if protected and not p.healthy():
            problems.append(f"the service at {c.health_url} is not healthy and not marked paused"
                            + (f" ({c.pause_file})" if c.pause_file else ""))
        level = p.pressure_level()
        if level != 1:
            problems.append(f"memory pressure level {level} (want 1 = normal)")
        fp = p.free_pct()
        free_gb = fp / 100.0 * c.total_gb
        if fp < 0 or free_gb < self.need_gb + c.reserve_gb:
            problems.append(f"only {free_gb:.1f} GB free ({fp}%); the engine needs {self.need_gb:.1f} GB "
                            f"+ {c.reserve_gb:.0f} GB reserve")
        if protected:
            others = []
            strata = p.processes("strata-inkling") - set(own_pids)
            if strata:
                others.append(f"strata-inkling pid {sorted(strata)}")
            llama = p.processes("llama-server")
            if llama:
                others.append(f"llama-server pid {sorted(llama)}")
            ol = p.ollama_models()
            if ol:
                others.append(f"Ollama model {', '.join(ol)} (`ollama stop <name>`)")
            if others:
                problems.append(f"the service at {c.health_url} is running and another large model is loaded: "
                                + "; ".join(others))
        return problems

    def arm(self) -> None:
        """Call just before the engine starts: the watch file's current count is the baseline."""
        self.watch_base = self.probes.watch_count() if self.cfg.watch_file else -1
        self.warn_since = None

    def check(self) -> str | None:
        """A reason to unload the engine now, or None."""
        p, c = self.probes, self.cfg
        if c.watch_file:
            n = p.watch_count()
            if self.watch_base >= 0 and n > self.watch_base:
                return f"{c.watch_file} logged a new '{c.watch_pattern}' ({self.watch_base} -> {n})"
        level = p.pressure_level()
        if level >= 4:
            return f"memory pressure level {level} (critical)"
        if level == 2:
            # A warning is normal for a few seconds while 17 GiB of weights load and the page cache is
            # evicted. Unload only if it does not clear within warn_grace_s.
            now = time.monotonic()
            if self.warn_since is None:
                self.warn_since = now
            elif now - self.warn_since >= c.warn_grace_s:
                return f"memory pressure level 2 for {now - self.warn_since:.0f}s"
        else:
            self.warn_since = None
        f = p.free_pct()
        if 0 <= f < c.floor_pct:
            return f"free memory {f}% < floor {c.floor_pct}%"
        return None
