import numpy as np
import torch


class IsaacPolicyBridge:
    """
    Isaac Sim 对接桥接层（骨架）：
    - 输入 Isaac 的 observation
    - 输出关节 action（当前实现为 BC 模型前向）
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

    @torch.no_grad()
    def action_from_observation(self, obs_vec: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(obs_vec.astype(np.float32)).unsqueeze(0).to(self.device)
        y = self.model(x).squeeze(0).cpu().numpy()
        return y
