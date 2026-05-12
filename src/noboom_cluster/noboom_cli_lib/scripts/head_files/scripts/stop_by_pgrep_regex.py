from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import subprocess
import time
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess:
    logger.debug("Running command: %s", " ".join(command))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    logger.debug("Command return code: %s", result.returncode)
    if stdout:
        logger.debug("Command stdout:\n%s", stdout)
    if stderr:
        logger.debug("Command stderr:\n%s", stderr)
    return result


def _pgrep(regex: str) -> list[int]:
    logger.info("Searching for processes with regex: %s", regex)
    result = _run_command(["pgrep", "-f", "--", regex])
    if result.returncode != 0:
        logger.info("No processes matched regex: %s", regex)
        return []
    pids = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            pids.append(int(line))
    logger.info("Matched PIDs: %s", pids if pids else "<none>")
    return pids


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        logger.debug("PID %s no longer exists.", pid)
        return False
    except PermissionError:
        logger.debug("PID %s exists but permission denied.", pid)
        return True
    return True


def _wait_gone(pids: Iterable[int], timeout_s: float, poll_interval_s: float) -> bool:
    start = time.time()
    pids = list(pids)
    logger.info(
        "Waiting for PIDs to exit: %s (timeout=%.2fs, poll=%.2fs)",
        pids,
        timeout_s,
        poll_interval_s,
    )
    while True:
        if not any(_pid_alive(pid) for pid in pids):
            logger.info("All PIDs have exited: %s", pids)
            return True
        if time.time() - start >= timeout_s:
            logger.warning("Timeout waiting for PIDs to exit: %s", pids)
            return False
        time.sleep(poll_interval_s)


def _tmux_available() -> bool:
    if _run_command(["which", "tmux"]).returncode != 0:
        logger.debug("tmux binary not found.")
        return False
    available = _run_command(["tmux", "ls"]).returncode == 0
    logger.debug("tmux available: %s", available)
    return available


def _tmux_panes() -> list[tuple[int, str]]:
    result = _run_command(["tmux", "list-panes", "-a", "-F", "#{pane_pid} #{session_name}"])
    panes = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        pane_pid_str, _, session_name = line.partition(" ")
        if pane_pid_str.isdigit() and session_name:
            panes.append((int(pane_pid_str), session_name))
    logger.debug("Discovered tmux panes: %s", panes)
    return panes


def _find_tmux_session_for_pid(pid: int, pane_map: dict[int, str]) -> Optional[str]:
    current = pid
    while True:
        if current <= 1:
            return None
        session = pane_map.get(current)
        if session:
            return session
        result = _run_command(["ps", "-o", "ppid=", "-p", str(current)])
        if result.returncode != 0:
            return None
        parent_str = (result.stdout or "").strip().splitlines()[0].strip() if result.stdout else ""
        if not parent_str.isdigit():
            return None
        current = int(parent_str)


def _collect_tmux_sessions(pids: Iterable[int]) -> list[str]:
    if not _tmux_available():
        return []
    panes = _tmux_panes()
    pane_map = {pane_pid: session for pane_pid, session in panes}
    sessions: set[str] = set()
    for pid in pids:
        session = _find_tmux_session_for_pid(pid, pane_map)
        if session:
            sessions.add(session)
    sessions_list = sorted(sessions)
    logger.info("Tmux sessions associated with PIDs: %s", sessions_list)
    return sessions_list


def _kill_tmux_sessions(sessions: Iterable[str]) -> None:
    for session in sessions:
        logger.info("Killing tmux session: %s", session)
        _run_command(["tmux", "kill-session", "-t", session])


def _kill_pids(pids: Iterable[int], sig: int) -> None:
    for pid in pids:
        try:
            logger.info("Sending signal %s to PID %s", sig, pid)
            os.kill(pid, sig)
        except ProcessLookupError:
            logger.debug("PID %s already exited.", pid)
            continue
        except PermissionError:
            logger.warning("Permission denied sending signal to PID %s", pid)
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop processes by pgrep regex")
    parser.add_argument("--pgrep-regex", required=True, type=str, help="Regex to match against pgrep output.")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument("--kill-after-timeout", action="store_true")
    parser.add_argument(
        "--tmux-session",
        dest="tmux_sessions",
        action="append",
        default=[],
        help="Named tmux session to kill after shutdown. Can be passed multiple times.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info(
        "Starting stop_by_pgrep_regex with regex=%s timeout=%.2fs poll=%.2fs kill_after_timeout=%s tmux_sessions=%s",
        args.pgrep_regex,
        args.timeout_s,
        args.poll_interval_s,
        args.kill_after_timeout,
        args.tmux_sessions,
    )

    requested_sessions = sorted(set(args.tmux_sessions))
    pids = _pgrep(args.pgrep_regex)
    sessions_to_kill = sorted(set(_collect_tmux_sessions(pids)) | set(requested_sessions))

    if not pids:
        if sessions_to_kill:
            _kill_tmux_sessions(sessions_to_kill)
        logger.info("No matching processes found; exiting.")
        return 0

    _kill_pids(pids, signal.SIGTERM)
    graceful = _wait_gone(pids, args.timeout_s, args.poll_interval_s)
    logger.info("Graceful shutdown completed: %s", graceful)

    if sessions_to_kill:
        _kill_tmux_sessions(sessions_to_kill)

    if args.kill_after_timeout and _pgrep(args.pgrep_regex):
        logger.warning("Processes still running; sending SIGKILL to matches.")
        _run_command(["pkill", "-KILL", "-f", "--", args.pgrep_regex])

    remaining = _pgrep(args.pgrep_regex)
    if remaining:
        logger.warning("Processes still running after shutdown attempts: %s", remaining)
        return 1
    logger.info("All matching processes stopped successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
