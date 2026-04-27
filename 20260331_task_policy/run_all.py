import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    root = Path(__file__).resolve().parent
    py = sys.executable
    run([str(py), "train_bc.py", "--config", "config.json", "--out", "outputs"], cwd=root)
    run([str(py), "train_act.py", "--config", "config.json", "--out", "outputs"], cwd=root)
    run([str(py), "analyze_results.py"], cwd=root)


if __name__ == "__main__":
    main()
