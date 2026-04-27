"""Compare BC and ACT-Chunk model outputs on the same observations."""
import sys, torch, numpy as np
sys.path.insert(0, r'e:\XT\vr-robot-control\20260409train2')

# Load BC model
bc_ckpt = torch.load(r'e:\XT\vr-robot-control\20260409train2\outputs\v4_opt\bc\run_20260412_150420\checkpoints\best.pt', map_location='cpu', weights_only=False)
bc_norm = bc_ckpt['normalizer']
act_mean = np.array(bc_norm['act_mean'], dtype=np.float32)
act_std = np.array(bc_norm['act_std'], dtype=np.float32)

from train_bc import TaskBCMLP
bc_cfg = bc_ckpt['config'].get('bc', {})
bc_model = TaskBCMLP(obs_dim=31, act_dim=8, hidden_dims=list(bc_cfg.get('hidden_dims', [256,256,128])), dropout=0.0)
bc_model.load_state_dict(bc_ckpt['model'])
bc_model.eval()

# Load ACT-Chunk model
ac_ckpt = torch.load(r'e:\XT\vr-robot-control\20260409train2\outputs\act_chunk_mujoco_lite\act_chunk\run_20260412_161809\checkpoints\best.pt', map_location='cpu', weights_only=False)
from train_act_chunk import ACTChunk
ac_model = ACTChunk(obs_dim=31, act_dim=8, chunk_size=5, d_model=128, nhead=4,
                     num_encoder_layers=3, num_decoder_layers=1, dim_feedforward=256, dropout=0.0, use_cvae=False)
ac_model.load_state_dict(ac_ckpt['model'])
ac_model.eval()

rng = np.random.default_rng(42)
bc_norms = []
ac_norms = []
for i in range(20):
    obs_z = rng.standard_normal(31).astype(np.float32)
    with torch.no_grad():
        bc_pred_z, _ = bc_model(torch.from_numpy(obs_z).unsqueeze(0))
        bc_raw = bc_pred_z.squeeze(0).numpy() * act_std + act_mean

        seq = np.zeros((10, 31), dtype=np.float32)
        seq[-1] = obs_z
        ac_out = ac_model(torch.from_numpy(seq).unsqueeze(0))
        ac_raw = ac_out['actions'].squeeze(0)[0].numpy() * act_std + act_mean

    bc_n = np.linalg.norm(bc_raw)
    ac_n = np.linalg.norm(ac_raw)
    bc_norms.append(bc_n)
    ac_norms.append(ac_n)
    if i < 5:
        print(f"Test {i}: BC_norm={bc_n:.6f}  AC_norm={ac_n:.6f}  ratio={ac_n/bc_n:.4f}")
        print(f"  BC: {np.round(bc_raw, 5)}")
        print(f"  AC: {np.round(ac_raw, 5)}")

print(f"\nAvg BC_norm: {np.mean(bc_norms):.6f}")
print(f"Avg AC_norm: {np.mean(ac_norms):.6f}")
print(f"Avg ratio (AC/BC): {np.mean(ac_norms)/np.mean(bc_norms):.4f}")
print(f"BC std_norm: {np.std(bc_norms):.6f}")
print(f"AC std_norm: {np.std(ac_norms):.6f}")
