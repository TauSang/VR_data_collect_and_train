import os, json, glob
base = '20260422_train/outputs/act_chunk'
for d in sorted(glob.glob(base + '/run_*')):
    cfg = os.path.join(d, 'config.json')
    ev = os.path.join(d, 'mujoco_eval.json')
    exp, sr = '?', '?'
    if os.path.exists(cfg):
        try:
            exp = json.load(open(cfg, encoding='utf-8')).get('experiment', '?')
        except Exception as e:
            exp = f'err:{e}'
    if os.path.exists(ev):
        try:
            e = json.load(open(ev, encoding='utf-8'))
            ok = e.get('total_success_overall') or e.get('success_count') or sum(t.get('total_success', 0) for t in e.get('trials', []))
            tot = e.get('total_targets_overall') or (len(e.get('trials', [])) * 5)
            if tot:
                sr = f'{ok}/{tot}={100*ok/tot:.0f}%'
        except Exception as ex:
            sr = f'err:{ex}'
    print(f'{os.path.basename(d):25s} {exp:40s} {sr}')
