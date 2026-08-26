import json
import sys

import numpy as np


def load(path):
    npz = np.load(path, allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    out = {"meta": meta}
    for act in ["linear", "relu"]:
        out[act] = {k: npz[f"{act}_{k}"] for k in [
            "checkpoints", "R2_train", "R2_train_stderr", "MI_train_nats",
            "R2_test", "R2_test_stderr", "MI_test_nats",
        ]}
    return out


for path in sys.argv[1:]:
    d = load(path)
    print(f"\n=== {path} (m={d['meta']['m']}, n={d['meta']['n']}, d={d['meta']['d']}) ===")
    for act in ["linear", "relu"]:
        r = d[act]
        t = r["checkpoints"]
        r2 = r["R2_train"]
        se = r["R2_train_stderr"]
        z = r2 / np.where(se > 0, se, np.nan)
        print(f"  [{act}] train R2 (z-score) at each checkpoint:")
        for tt, rr, ss, zz in zip(t, r2, se, z):
            print(f"    t={tt:6d}  R2={rr:+.3e}  stderr={ss:.3e}  z={zz:+.2f}")
        print(f"  [{act}] final: R2_train={r2[-1]:+.3e}, R2_test={r['R2_test'][-1]:+.3e}, "
              f"gap={r2[-1]-r['R2_test'][-1]:+.3e}")
