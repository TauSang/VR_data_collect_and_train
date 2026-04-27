import subprocess
from pathlib import Path


def run(cmd, cwd):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    root = Path(__file__).resolve().parent
    py = root.parents[1] / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path("python")

    run([str(py), "train_bc.py", "--config", "config.json", "--out", "outputs"], cwd=root)
    run([str(py), "train_act.py", "--config", "config.json", "--out", "outputs"], cwd=root)
    run([str(py), "analyze_results.py"], cwd=root)


if __name__ == "__main__":
    main()
