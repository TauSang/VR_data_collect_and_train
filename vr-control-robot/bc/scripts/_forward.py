from pathlib import Path
import runpy
import sys


def forward_to_root_script(script_name: str):
    root = Path(__file__).resolve().parents[2]
    target = root / "scripts" / script_name
    if not target.exists():
        raise FileNotFoundError(f"未找到目标脚本: {target}")

    # 确保相对导入可解析
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    runpy.run_path(str(target), run_name="__main__")
