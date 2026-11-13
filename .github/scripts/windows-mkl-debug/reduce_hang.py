"""Reduce the Windows PARDISO/tomography hang using fresh pytest processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import pytest
    import qutip

    tests = Path(qutip.__file__).parent / "tests"

    class Recorder:
        def pytest_collection_finish(self, session):
            if args.collect:
                nodes = [
                    str(item.path) + "::" + item.nodeid.split("::", 1)[1]
                    for item in session.items
                ]
                Path(args.collect).write_text(json.dumps(nodes))

        def pytest_runtest_logstart(self, nodeid, location):
            print("START_NODE " + nodeid, flush=True)

    if args.collect:
        options = ["--collect-only", "-q", str(tests / "test_mkl.py")]
    else:
        nodes = json.loads(Path(args.nodes).read_text())
        options = ["-vv", "-s", "--timeout=120", "--timeout-method=thread"]
        options += nodes + [str(tests / "test_qpt.py") + "::test_qpt_cnot"]
    return int(pytest.main(options, plugins=[Recorder()]))


def reduce(args):
    logs = Path(args.logs).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    collection = logs / "collected-nodes.json"
    with (logs / "collection.log").open("w") as output:
        subprocess.run(
            [sys.executable, "-u", script, "--collect", str(collection)],
            stdout=output, stderr=subprocess.STDOUT, check=True, timeout=120,
        )
    nodes = json.loads(collection.read_text())
    if not nodes:
        raise RuntimeError("No MKL tests collected; check pydiso availability")
    deadline = time.monotonic() + args.budget
    results = []

    def attempt(label, selected):
        remaining = deadline - time.monotonic()
        if remaining < 140:
            raise TimeoutError("Reduction budget exhausted; inspect saved results")
        number = len(results)
        nodefile = logs / f"{number:03d}-{label}-nodes.json"
        logfile = logs / f"{number:03d}-{label}.log"
        nodefile.write_text(json.dumps(selected, indent=2))
        command = [sys.executable, "-u", script, "--nodes", str(nodefile)]
        with logfile.open("w") as output:
            try:
                completed = subprocess.run(
                    command, stdout=output, stderr=subprocess.STDOUT, timeout=140,
                )
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = None
        content = logfile.read_text(errors="replace")
        # Pytest's progress output may precede the hook marker on the same line.
        starts = [line.split("START_NODE ", 1)[1].strip()
                  for line in content.splitlines() if "START_NODE " in line]
        at_tomography = bool(starts and starts[-1].endswith("test_qpt.py::test_qpt_cnot"))
        timed_out = code is None or "Timeout >" in content
        outcome = "hang" if timed_out and at_tomography else "pass" if code == 0 else "error"
        results.append(dict(label=label, nodes=selected, outcome=outcome,
                            returncode=code, log=logfile.name, command=command))
        (logs / "results.json").write_text(json.dumps(results, indent=2))
        print(f"{label}: {outcome} ({len(selected)} MKL cases); {logfile.name}", flush=True)
        if outcome == "error":
            raise RuntimeError(f"Unexpected failure; inspect {logfile}")
        return outcome == "hang"

    def report(message, selected=None):
        if selected is not None:
            (logs / "reproducer-nodes.json").write_text(json.dumps(selected, indent=2))
            message += "\n\nMKL cases preceding test_qpt_cnot:\n\n"
            message += "\n".join(f"- `{node}`" for node in selected)
        (logs / "summary.md").write_text(message + "\n")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
                output.write(message + "\n")
        print(message, flush=True)

    try:
        if attempt("tomography-alone", []):
            report("Tomography hangs alone; interaction baseline not reproduced.")
            return 1
        if not attempt("full-baseline", nodes):
            report("Full MKL module followed by test_qpt_cnot passed; no reduction performed.")
            return 1
        for node in nodes:
            if attempt("individual", [node]):
                report("Found an individual MKL case that triggers the tomography hang.", [node])
                return 0

        # Delta debugging: try chunks and their complements, preserving order.
        candidate = nodes
        granularity = 2
        while len(candidate) >= 2:
            chunks = [candidate[i * len(candidate) // granularity:
                                (i + 1) * len(candidate) // granularity]
                      for i in range(granularity)]
            subsets = chunks + [[node for node in candidate if node not in chunk]
                                for chunk in chunks]
            for subset in subsets:
                if subset and len(subset) < len(candidate) and attempt("group", subset):
                    candidate = subset
                    granularity = max(2, granularity - 1)
                    break
            else:
                if granularity == len(candidate):
                    break
                granularity = min(len(candidate), granularity * 2)
                continue
        report("Reduced to a group with no further reduction found; repeat to check reproducibility.", candidate)
        return 0
    except (TimeoutError, RuntimeError) as error:
        report(str(error) + "; results.json contains completed comparisons.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect")
    parser.add_argument("--nodes")
    parser.add_argument("--logs", default="mkl-debug-logs/reduction")
    parser.add_argument("--budget", type=int, default=1080)
    arguments = parser.parse_args()
    sys.exit(worker(arguments) if arguments.collect or arguments.nodes else reduce(arguments))
