"""Resumable orchestrator for all method-specific D2 trajectories."""

import argparse
import subprocess
import sys

from phase_d2_common import CONFIG
from project_paths import ROOT


parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=CONFIG["methods"])
args = parser.parse_args()
methods = [args.method] if args.method else CONFIG["methods"]


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / "scripts_local" / script), *arguments], check=True)


for method in methods:
    for round_index in range(CONFIG["allocation_rounds"]):
        common = ["--method", method, "--round", str(round_index)]
        run("phase_d2_policy.py", *common)
        run("phase_d2_prm.py", *common)
        run("phase_d2_allocate.py", *common)
    run(
        "phase_d2_policy.py", "--method", method,
        "--round", str(CONFIG["allocation_rounds"]), "--final",
    )
print("D2 generation complete")
