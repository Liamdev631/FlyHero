"""Does a hand-coded rule recover the strum label from the observation?

If a two-line rule scores near-perfectly, then the dataset is a strawman: the
answer is already present in the features, so a network failing at it means the
training setup (or the formulation) is the problem - not the task's difficulty.
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "/home/liamb/projects/FlyHero/tools")
from flyhero import ann as A  # noqa: E402

STEP = 1.0 / 100.0  # controller step, 10 ms
NET_F1 = {
    "FlyHero_Test_One_Easy": 0.464,
    "FlyHero_Test_Three_Easy": 0.074,
    "FlyHero_Test_Two_Easy": 0.054,
}

print("Rule: strum when the min lane delta crosses INTO one step (needs frame stacking)\n")
print(f"{'song':<28} {'rule F1':>8} {'P':>7} {'R':>7}   {'net F1':>7}")

tot_tp = tot_fp = tot_fn = 0
for path in sorted(glob.glob("/home/liamb/projects/FlyHero-build/automation-data/datasets/*.npz")):
    d = np.load(path)
    x, strum = d["x"], d["strum"]

    cur = x[:, 0:10:2].min(axis=1)     # current-frame min delta over the 5 lanes
    prev = x[:, 11:21:2].min(axis=1)   # previous frame (stack=2)

    crossed = np.logical_and(cur <= STEP, prev > STEP)
    pred = crossed.astype(np.float32)

    m = A.score(strum, pred, 0.5)
    key = path.split("/")[-1].replace(".npz", "")
    print(f"{key:<28} {m.f1:>8.3f} {m.precision:>7.3f} {m.recall:>7.3f}   {NET_F1.get(key, 0):>7.3f}")

    tot_tp += m.tp
    tot_fp += m.fp
    tot_fn += m.fn

agg = A.Metrics(tot_tp, tot_fp, tot_fn, 0)
print(f"\n{'ALL SONGS':<28} {agg.f1:>8.3f} {agg.precision:>7.3f} {agg.recall:>7.3f}")
