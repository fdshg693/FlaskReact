#!/usr/bin/env python3
"""
.procmanファイルを利用して、バックグラウンドでコマンドを実行するシンプルなプロセスマネージャ。

Usage:
    python procman.py run <cmd>   - Run command in background
    python procman.py list        - List managed processes
    python procman.py kill <id>   - Kill process by ID
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path.home() / ".procman"
DATA_FILE = DATA_DIR / "processes.json"


def load_processes():
    """Load processes from JSON file."""
    if not DATA_FILE.exists():
        return {}
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_processes(processes):
    """Save processes to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(processes, f, indent=2)


def get_next_id(processes):
    """Get the next available ID."""
    if not processes:
        return 1
    return max(int(k) for k in processes.keys()) + 1  # type: ignore


def is_alive(pid):
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def cmd_run(cmd):
    """Run a command in the background."""
    devnull = open(os.devnull, "w")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=devnull, stderr=devnull, start_new_session=True
    )

    processes = load_processes()
    new_id = get_next_id(processes)
    processes[str(new_id)] = {
        "pid": proc.pid,
        "cmd": cmd,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    save_processes(processes)

    print(f"[{new_id}] Started (PID: {proc.pid})")


def cmd_list():
    """List all managed processes."""
    processes = load_processes()

    if not processes:
        print("No managed processes.")
        return

    print(f"{'ID':<5}{'PID':<8}{'STATUS':<10}COMMAND")
    for id_, info in sorted(processes.items(), key=lambda x: int(x[0])):
        pid = info["pid"]
        status = "running" if is_alive(pid) else "dead"
        cmd = info["cmd"]
        print(f"{id_:<5}{pid:<8}{status:<10}{cmd}")


def cmd_kill(id_):
    """Kill a process by ID."""
    processes = load_processes()

    if id_ not in processes:
        print(f"Error: ID {id_} not found.")
        sys.exit(1)

    pid = processes[id_]["pid"]

    try:
        os.kill(pid, 15)  # SIGTERM
    except (OSError, ProcessLookupError):
        pass  # Process already dead

    del processes[id_]
    save_processes(processes)

    print(f"[{id_}] Killed")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "run":
        if len(sys.argv) < 3:
            print("Error: run requires a command argument.")
            sys.exit(1)
        cmd_run(sys.argv[2])

    elif command == "list":
        cmd_list()

    elif command == "kill":
        if len(sys.argv) < 3:
            print("Error: kill requires an ID argument.")
            sys.exit(1)
        cmd_kill(sys.argv[2])

    else:
        print(f"Error: Unknown command '{command}'.")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
