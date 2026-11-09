# Test RMSE vs training-set size. Needs run_seeds.py in the same folder.
import rydberg_core as rc
from rydberg_core import *
import sys; sys.argv=['x']
# reuse the model code from run_seeds.py (everything before its main loop)
exec(open('run_seeds.py').read().split("rows = []")[0].split("import rydberg_core as rc")[1].split("from rydberg_core import *")[1])
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
p_te, y_te, _ = make_set(N_TEST, 2000); _, F = feats(p_te, "M1"); rows = []
m0 = rmse(F["logRq"].values, y_te)
# one big training set; each smaller training set is its first n pulses
p_all, y_all, _ = make_set(10000, 1000)
for n in [250, 500, 1000, 3000, 10000]:
    m = fit_all(p_all[:n], y_all[:n])
    r = dict(n_train=n, M0=m0, **{k: rmse(pred(m, p_te, k), y_te) for k in ["M1", "M2", "M3"]}); rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/learning_curve.csv", index=False)
