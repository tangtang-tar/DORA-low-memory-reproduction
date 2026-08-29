"""C2 总控：每个模型使用独立子进程，退出后释放 8 GB 显存。"""

import subprocess
import sys
from pathlib import Path

from phase_c2_common import CONFIG
from project_paths import ROOT


SCRIPT_DIR = ROOT / "scripts_local"


def run(script, *arguments):
    command = [sys.executable, str(SCRIPT_DIR / script), *arguments]
    subprocess.run(command, check=True)


for round_index in range(CONFIG["allocation_rounds"]):
    run("phase_c2_policy_round.py", "--round", str(round_index))
    run("phase_c2_prm_round.py", "--round", str(round_index))
    run("phase_c2_allocate_round.py", "--round", str(round_index))

run(
    "phase_c2_policy_round.py",
    "--round",
    str(CONFIG["allocation_rounds"]),
    "--final",
)
print("C2 多轮生成完成")
