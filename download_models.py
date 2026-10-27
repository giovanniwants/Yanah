"""
Resumable, verified model download for Yanah.

The Hugging Face repo (made by publish_models.py) holds:

    yanah_manifest.json                 every file below with its size + sha256
    dense/                              config, tokenizer, model.safetensors
                                        (only the non-expert weights: 1.3 GiB)
    experts-2bit/experts.json
    experts-2bit/parts/layer-000.bin    one file per layer, 576 MiB each,
    ...                                 48 files = 27 GiB
    experts-4bit/...                    (optional, 41 GiB)

Why the experts come in 48 parts instead of one 27 GiB experts.bin
------------------------------------------------------------------
* A corrupt or truncated file costs one 576 MiB part, not 27 GiB.
* Each part is verified (sha256) the moment it lands, then copied into its
  place in experts.bin and deleted, so the download never needs more than
  ~1 GiB of disk beyond the final 28 GiB.
* Progress survives anything: quitting, sleep, Wi-Fi drops, a crash. Done
  parts are recorded in a state file; the part in flight resumes from the
  byte it reached (huggingface_hub keeps a .incomplete file and continues it
  with an HTTP range request; resume_download=True was removed from
  huggingface_hub because resuming is now always on).

Connection drops are retried forever with backoff (up to 60 s between
tries) until the user cancels; problems that retrying can't fix (repo
missing, disk full) stop with a plain message.

    python download_models.py                 # download / resume (2-bit set)
    python download_models.py --verify        # re-hash everything, re-fetch bad parts
"""
from __future__ import annotations

import argparse
import collections
import errno
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yanah_runtime as rt

MANIFEST = "yanah_manifest.json"
STAGING = ".yanah_download"
LEAVE_FREE_GB = 5.0            # never fill the disk past this
HASH_CHUNK = 8 << 20
RATE_WINDOW_S = 10.0          # download speed is measured over the last 10 seconds
RATE_EMA_ALPHA = 0.2          # then smoothed (exponential moving average), updated once a second
ETA_MAX_S = 99 * 3600         # never show more than this (no "infinity")


class DownloadError(Exception):
    """A problem retrying won't fix. `str(e)` is written for the user."""


class Cancelled(Exception):
    pass


@dataclass
class Progress:
    phase: str = "starting"                 # starting|manifest|dense|experts|verifying|done|error|waiting
    done_bytes: int = 0
    total_bytes: int = 0
    message: str = ""
    file: str = ""
    rate_bps: float = 0.0                   # smoothed speed (10 s window + EMA)
    eta_s: Optional[float] = None           # last good time-remaining estimate, seconds
    eta_text: str = "Calculating..."        # what to show; never empty, never infinite

    @property
    def fraction(self) -> float:
        return self.done_bytes / self.total_bytes if self.total_bytes else 0.0


def sha256_file(path: Path, offset: int = 0, length: Optional[int] = None,
                cancel: Optional[threading.Event] = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        left = length if length is not None else (os.path.getsize(path) - offset)
        while left > 0:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            b = f.read(min(HASH_CHUNK, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


# ----------------------------------------------------------------- the hub
class HubClient:
    """The two things the downloader needs from Hugging Face. Tests swap in
    a fake with the same methods."""

    def __init__(self, repo_id: str, revision: Optional[str] = None):
        self.repo_id, self.revision = repo_id, revision

    def download(self, filename: str, local_dir: Path, force: bool = False) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(self.repo_id, filename, revision=self.revision,
                                    local_dir=str(local_dir), force_download=force))

    def remote_sha256(self, paths: List[str]) -> Dict[str, str]:
        """sha256 the repository itself records for each LFS file (best effort)."""
        from huggingface_hub import HfApi
        out = {}
        api = HfApi()
        for i in range(0, len(paths), 50):
            for f in api.get_paths_info(self.repo_id, paths[i:i + 50], revision=self.revision):
                lfs = getattr(f, "lfs", None)
                sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
                if sha:
                    out[f.path] = sha
        return out


def classify(e: BaseException) -> str:
    """'permanent' or 'transient'."""
    name = type(e).__name__
    if isinstance(e, (DownloadError, Cancelled)):
        return "permanent"
    if name in ("RepositoryNotFoundError", "GatedRepoError", "RevisionNotFoundError",
                "EntryNotFoundError", "RemoteEntryNotFoundError", "DisabledRepoError"):
        return "permanent"
    if isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES):
        return "permanent"
    resp = getattr(e, "response", None)
    code = getattr(resp, "status_code", None)
    if code in (401, 403, 404):
        return "permanent"
    return "transient"            # connection drops, timeouts, 5xx, 429, offline


def friendly(e: BaseException) -> str:
    name = type(e).__name__
    if isinstance(e, DownloadError):
        return str(e)
    if name in ("RepositoryNotFoundError", "GatedRepoError", "DisabledRepoError") or \
            getattr(getattr(e, "response", None), "status_code", None) in (401, 403, 404):
        return ("Yanah's download source isn't available (the model repository is missing or private). "
                "Check the model_repo setting.")
    if isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT):
        return "Your disk is full. Free up space and press Start Download again; progress is kept."
    if isinstance(e, OSError) and e.errno in (errno.EROFS, errno.EACCES):
        return f"Yanah can't write to {getattr(e, 'filename', 'the models folder')}."
    return f"{name}: {e}"


# -------------------------------------------------------------- downloader
@dataclass
class _State:
    path: Path
    data: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "_State":
        try:
            return cls(path, json.loads(path.read_text()))
        except Exception:
            return cls(path, {})

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, self.path)


class Downloader:
    def __init__(self, repo_id: str, dest_root: Path, experts: str = "2bit",
                 on_progress: Optional[Callable[[Progress], None]] = None,
                 hub=None, cancel: Optional[threading.Event] = None,
                 max_backoff_s: float = 60.0, sleep=time.sleep):
        self.repo_id, self.root, self.experts = repo_id, Path(dest_root), experts
        self.hub = hub or HubClient(repo_id)
        self.cancel = cancel or threading.Event()
        self.on_progress = on_progress or (lambda p: None)
        self.max_backoff_s, self.sleep = max_backoff_s, sleep
        self.p = Progress()
        self.staging = self.root / STAGING
        self.dense_dir = self.root / "Qwen3-80B-dense"
        self.exp_dir = self.root / rt.EXPERT_DIRS[experts]
        # speed tracking: (time, bytes) samples over the last RATE_WINDOW_S seconds
        self._samples: "collections.deque[tuple[float, int]]" = collections.deque()
        self._rate_key = None          # (phase is verifying, total_bytes): reset tracking when it changes
        self._max_done = 0             # bytes only count forward (the in-flight count dips between files)
        self._last_rate_t = 0.0

    # -- reporting
    def _report(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.p, k, v)
        self._update_rate(time.monotonic())   # monotonic: immune to clock changes
        self.on_progress(self.p)

    @staticmethod
    def _fmt_eta(s: float) -> str:
        if s > 5400:
            return f"about {s / 3600:.1f} hours left"
        if s >= 90:
            return f"about {s / 60:.0f} min left"
        return "less than 2 min left"

    def _update_rate(self, now: float) -> None:
        """Smoothed speed and time estimate. The speed is measured over the last 10 seconds
        and then averaged (EMA), so one slow or fast moment can't swing the estimate."""
        key = (self.p.phase == "verifying", self.p.total_bytes)
        if key != self._rate_key:                     # new job (download vs. file check): start fresh
            self._rate_key = key
            self._samples.clear()
            self._max_done = self.p.done_bytes
            self._last_rate_t = 0.0
            self.p.rate_bps, self.p.eta_s, self.p.eta_text = 0.0, None, "Calculating..."
        self._max_done = max(self._max_done, self.p.done_bytes)
        if now - self._last_rate_t < 1.0:
            return
        self._last_rate_t = now
        self._samples.append((now, self._max_done))
        while len(self._samples) > 1 and now - self._samples[0][0] > RATE_WINDOW_S:
            self._samples.popleft()
        t0, b0 = self._samples[0]
        if now - t0 < 1.0:
            return                                    # not enough history yet: keep "Calculating..."
        window = (self._max_done - b0) / (now - t0)  # bytes/s over the last ~10 s
        if window <= 0:
            # Stalled: keep the last known estimate and say we're recalculating.
            self.p.eta_text = (f"{self._fmt_eta(self.p.eta_s)} · Calculating..."
                               if self.p.eta_s is not None else "Calculating...")
            return
        self.p.rate_bps = window if self.p.rate_bps <= 0 else \
            RATE_EMA_ALPHA * window + (1 - RATE_EMA_ALPHA) * self.p.rate_bps
        remaining = max(0, self.p.total_bytes - self._max_done)
        self.p.eta_s = min(ETA_MAX_S, remaining / self.p.rate_bps)
        self.p.eta_text = self._fmt_eta(self.p.eta_s)

    def _in_flight_bytes(self) -> int:
        """Bytes of the file being downloaded right now (huggingface_hub's .incomplete)."""
        n = 0
        cache = self.staging / ".cache" / "huggingface" / "download"
        if cache.is_dir():
            for f in cache.rglob("*.incomplete"):
                try:
                    n += f.stat().st_size
                except OSError:
                    pass
        return n

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    # -- retry
    def _retry(self, what: str, fn):
        delay, attempt = 2.0, 0
        unknown = 0
        while True:
            self._check_cancel()
            try:
                return fn()
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                if classify(e) == "permanent":
                    raise
                name = type(e).__name__
                if name not in ("ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError", "WriteError",
                                "RemoteProtocolError", "LocalEntryNotFoundError", "HfHubHTTPError",
                                "ConnectionError", "ConnectionResetError", "TimeoutError", "ProxyError",
                                "PoolTimeout", "NetworkError", "TransportError", "ChunkedEncodingError"):
                    unknown += 1
                    if unknown > 5:          # a bug, not the network: don't loop forever
                        raise
                attempt += 1
                prev_phase = self.p.phase
                for left in range(int(delay), 0, -1):
                    self._check_cancel()
                    self._report(phase="waiting",
                                 message=f"Connection problem while getting {what}. Retrying in {left}s "
                                         f"(attempt {attempt}). Your progress is saved.")
                    self.sleep(1)
                self._report(phase=prev_phase, message=f"Resuming {what}…")
                delay = min(self.max_backoff_s, delay * 2)

    # -- steps
    def _fetch(self, repo_path: str, expect_sha: str, expect_size: int) -> Path:
        """Download one repo file into staging, verify it, retrying a corrupt
        copy (fresh download) up to 3 times."""
        for attempt in range(3):
            force = attempt > 0
            local = self._retry(repo_path, lambda: self.hub.download(repo_path, self.staging, force=force))
            self._check_cancel()
            size = local.stat().st_size
            if size == expect_size:
                self._report(message=f"Checking {Path(repo_path).name}…")
                if sha256_file(local, cancel=self.cancel) == expect_sha:
                    return local
            local.unlink(missing_ok=True)
            self._report(message=f"{Path(repo_path).name} arrived damaged; downloading it again.")
        raise DownloadError(f"{repo_path} keeps arriving damaged (checksum mismatch). "
                            "The download source may be broken.")

    def _load_manifest(self) -> dict:
        self._report(phase="manifest", message="Contacting Hugging Face…")
        path = self._retry("the file list", lambda: self.hub.download(MANIFEST, self.staging, force=True))
        m = json.loads(Path(path).read_text())
        if self.experts not in m.get("experts", {}):
            raise DownloadError(f"The download source has no {self.experts} experts.")
        return m

    def _expected(self, manifest: dict) -> Dict[str, str]:
        files = manifest["dense"]["files"] + manifest["experts"][self.experts]["parts"] + \
                [manifest["experts"][self.experts]["json"]]
        want = {f["path"]: f["sha256"] for f in files}
        try:                      # what the repository itself says wins over the manifest
            remote = self._retry("checksums", lambda: self.hub.remote_sha256(sorted(want)))
            for k, v in remote.items():
                if k in want and v != want[k]:
                    want[k] = v
        except DownloadError:
            raise
        except Exception:
            pass
        return want

    def _preflight(self, need: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.root).free
        need_all = need + 1.2 * 2**30 + LEAVE_FREE_GB * 2**30
        if free < need_all:
            raise DownloadError(
                f"Yanah needs {need / 1e9:.1f} GB more disk space (plus {LEAVE_FREE_GB:.0f} GB kept free for macOS); "
                f"this Mac has {free / 1e9:.1f} GB free. Free up space and press Start Download again.")

    def run(self) -> Dict[str, Path]:
        self.staging.mkdir(parents=True, exist_ok=True)
        state = _State.load(self.staging / f"state-{self.experts}.json")
        m = self._load_manifest()
        want = self._expected(m)
        dense_files = m["dense"]["files"]
        ex = m["experts"][self.experts]
        parts, layer_bytes = ex["parts"], ex["layer_bytes"]
        total = len(parts) * layer_bytes
        final_bin = self.exp_dir / "experts.bin"

        dense_done = [f for f in dense_files if self._dense_file_ok(f, want, state)]
        bin_done = rt.experts_present(self.exp_dir)
        parts_done = set(range(len(parts))) if bin_done else set(state.data.get("parts_done", []))
        all_bytes = sum(f["size"] for f in dense_files) + total
        have = sum(f["size"] for f in dense_done) + len(parts_done) * layer_bytes
        self._report(total_bytes=all_bytes, done_bytes=have)
        self._preflight(all_bytes - have)

        # 1. dense weights + tokenizer (small)
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        for f in dense_files:
            if f in dense_done:
                continue
            self._report(phase="dense", file=f["path"], message="Downloading Yanah's language core…")
            got = self._fetch_tracked(f["path"], want[f["path"]], f["size"])
            target = self.dense_dir / Path(f["path"]).relative_to("dense")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(got), str(target))
            st = target.stat()
            state.data.setdefault("dense_ok", {})[f["path"]] = [st.st_size, st.st_mtime_ns, want[f["path"]]]
            state.save()
            self._report(done_bytes=self.p.done_bytes + f["size"])

        # 2. experts: parts land straight into their slot of experts.bin.partial
        if not bin_done:
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            partial = self.exp_dir / "experts.bin.partial"
            mode = "r+b" if partial.exists() else "w+b"
            with open(partial, mode) as out:
                if os.path.getsize(partial) != total:
                    out.truncate(total)                 # sparse on APFS: no space used until written
                for i, part in enumerate(parts):
                    if i in parts_done:
                        continue
                    self._report(phase="experts", file=part["path"],
                                 message=f"Downloading Yanah's experts: part {i + 1} of {len(parts)}…")
                    got = self._fetch_tracked(part["path"], want[part["path"]], part["size"])
                    with open(got, "rb") as src:
                        out.seek(i * layer_bytes)
                        shutil.copyfileobj(src, out, 16 << 20)
                    out.flush()
                    os.fsync(out.fileno())
                    got.unlink(missing_ok=True)
                    parts_done.add(i)
                    state.data["parts_done"] = sorted(parts_done)
                    state.save()
                    self._report(done_bytes=self.p.done_bytes + layer_bytes)
            jf = ex["json"]
            got = self._fetch(jf["path"], want[jf["path"]], jf["size"])
            shutil.move(str(got), str(self.exp_dir / "experts.json"))
            os.replace(partial, final_bin)
            if not rt.experts_present(self.exp_dir):
                raise DownloadError("experts.bin has the wrong size after download; press Verify to repair.")
        state.data.pop("parts_done", None)
        state.save()
        self._report(phase="done", done_bytes=all_bytes, message="Download complete.")
        return {"dense": self.dense_dir, "experts": self.exp_dir}

    def _fetch_tracked(self, path: str, sha: str, size: int) -> Path:
        """_fetch with live byte progress from the in-flight .incomplete file."""
        base = self.p.done_bytes
        stop = threading.Event()

        def tick():
            while not stop.wait(0.5):
                self._report(done_bytes=base + min(size, self._in_flight_bytes()))
        t = threading.Thread(target=tick, daemon=True)
        t.start()
        try:
            return self._fetch(path, sha, size)
        finally:
            stop.set()
            t.join()
            self._report(done_bytes=base)

    def _dense_file_ok(self, f: dict, want: Dict[str, str], state: _State) -> bool:
        target = self.dense_dir / Path(f["path"]).relative_to("dense")
        # a full 4-bit checkpoint elsewhere doesn't count: we only look at our own folder
        if not target.is_file() or target.stat().st_size != f["size"]:
            return False
        rec = state.data.get("dense_ok", {}).get(f["path"])
        st = target.stat()
        return bool(rec) and rec == [st.st_size, st.st_mtime_ns, want[f["path"]]]

    # -- repair
    def verify(self) -> List[str]:
        """Re-hash the installed files against the source. Damaged expert
        layers are marked for re-download; returns what was bad. Run run()
        afterwards to fetch them."""
        self.staging.mkdir(parents=True, exist_ok=True)
        m = self._load_manifest()
        want = self._expected(m)
        ex = m["experts"][self.experts]
        parts, layer_bytes = ex["parts"], ex["layer_bytes"]
        state = _State.load(self.staging / f"state-{self.experts}.json")
        bad: List[str] = []
        total = sum(f["size"] for f in m["dense"]["files"]) + len(parts) * layer_bytes
        self._report(phase="verifying", total_bytes=total, done_bytes=0, message="Checking Yanah's files…")
        for f in m["dense"]["files"]:
            target = self.dense_dir / Path(f["path"]).relative_to("dense")
            if not target.is_file() or target.stat().st_size != f["size"] or \
                    sha256_file(target, cancel=self.cancel) != want[f["path"]]:
                bad.append(f["path"])
                state.data.get("dense_ok", {}).pop(f["path"], None)
                target.unlink(missing_ok=True)
            self._report(done_bytes=self.p.done_bytes + f["size"])
        b = self.exp_dir / "experts.bin"
        partial = self.exp_dir / "experts.bin.partial"
        src = b if b.exists() else partial
        good = set()
        if src.exists() and src.stat().st_size == len(parts) * layer_bytes:
            done_before = set(range(len(parts))) if src == b else set(state.data.get("parts_done", []))
            for i, part in enumerate(parts):
                self._report(message=f"Checking experts part {i + 1} of {len(parts)}…")
                if i in done_before and sha256_file(src, i * layer_bytes, layer_bytes, self.cancel) == want[part["path"]]:
                    good.add(i)
                elif i in done_before:
                    bad.append(part["path"])
                self._report(done_bytes=self.p.done_bytes + layer_bytes)
            if src == b and len(good) < len(parts):
                os.replace(b, partial)           # reopen for repair
        else:
            bad.append("experts.bin (missing or wrong size)")
            if src.exists():
                src.unlink()
        state.data["parts_done"] = sorted(good)
        state.save()
        self._report(phase="done", message="All files are intact." if not bad else
                     f"{len(bad)} damaged file(s) will be downloaded again.")
        return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = rt.load_settings()
    ap.add_argument("--repo", default=rt.model_repo(s))
    ap.add_argument("--dest", default=str(rt.download_root()))
    ap.add_argument("--experts", choices=sorted(rt.EXPERT_DIRS), default=rt.wanted_download(s))
    ap.add_argument("--verify", action="store_true", help="re-hash installed files and repair damaged ones")
    a = ap.parse_args(argv)
    last = [0.0]

    def show(p: Progress):
        if time.time() - last[0] < 0.5 and p.phase not in ("done", "waiting"):
            return
        last[0] = time.time()
        sys.stderr.write(f"\r{p.fraction:6.1%} {p.done_bytes / 1e9:6.2f}/{p.total_bytes / 1e9:.2f} GB "
                         f"{p.rate_bps / 1e6:6.1f} MB/s  {p.eta_text:<28} {p.message[:60]:<60}")
    d = Downloader(a.repo, Path(a.dest).expanduser(), a.experts, on_progress=show)
    try:
        if a.verify:
            bad = d.verify()
            print(f"\n{len(bad)} damaged: {bad}" if bad else "\nall files intact", file=sys.stderr)
        d.run()
        print(f"\nmodels ready in {a.dest}", file=sys.stderr)
    except (DownloadError, Cancelled) as e:
        raise SystemExit(f"\n{friendly(e)}")
    except Exception as e:
        raise SystemExit(f"\n{friendly(e)}")


if __name__ == "__main__":
    main()
