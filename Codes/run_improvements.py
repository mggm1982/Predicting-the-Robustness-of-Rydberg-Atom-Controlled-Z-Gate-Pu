# Shift tests, search baselines and timing. Usage: PRESET=realistic|stress OUT=<folder> PART=OOD|SEARCH|TIME python run_improvements.py
import rydberg_core as rc
from rydberg_core import *
import importlib
# which experiment to run: OOD (shifted pulses), SEARCH (search baselines) or TIME (timing)
PART = os.environ.get("PART", "OOD")

# same models as in run_pipeline.py
class Ens:
    def __init__(self, k=5, seed=0): self.k, self.seed, self.m = k, seed, []
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed); self.m = []
        for i in range(self.k):
            idx = rng.integers(0, len(X), len(X))
            self.m.append(HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                          l2_regularization=1e-3, random_state=i).fit(X.iloc[idx], y[idx]))
        return self
    def predict(self, X):
        P = np.array([m.predict(X) for m in self.m]); return P.mean(0), P.std(0)
def feats(p, kind):
    X, F = raw_features(p), phys_features(p)
    return {"M1": X, "M2": pd.concat([X, F], axis=1), "M3": pd.concat([X, F], axis=1)}[kind], F
def fit_all(p, y):
    out = {}
    for k in ["M1", "M2", "M3"]:
        X, F = feats(p, k); out[k] = Ens().fit(X, y - F["logRq"].values if k == "M3" else y)
    return out
def pred(models, p, k):
    X, F = feats(p, k); mu, sd = models[k].predict(X); return (mu + F["logRq"].values if k == "M3" else mu), sd
# conformal quantile, as in run_pipeline.py
def cq(y, mu, sd, alpha=ALPHA):
    s = np.abs(y - mu) / (sd + 1e-3); n = len(s); return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

# OOD: train as usual, then test on pulses made in other ways than the training pulses
# and see how the errors and the interval coverage change
if PART == "OOD":
    p_tr, y_tr, _ = make_set(N_TRAIN, 101); p_ca, y_ca, _ = make_set(N_CAL, 202)
    models = fit_all(p_tr, y_tr)
    rng = np.random.default_rng(55); n = N_TEST
    # test sets: same generator, uniform random pulses, short hill-climb, near the published pulse, long pulses
    sets = {"same generator (reference)": make_set(n, 303)[0],
            "uniform random pulses": rng.uniform(LO, HI, (n, 5)),
            "short hill-climb (40 it.)": gen_pulses(n, 5001, iters=40),
            "near published pulse": np.clip(PUB[0] + 0.03 * (HI - LO) * rng.normal(size=(n, 5)), LO, HI),
            "long pulses only (T>=12)": None}
    pp = gen_pulses(4 * n, 5002); sets["long pulses only (T>=12)"] = pp[pp[:, 4] >= 12][:n]
    # the interval size q is fixed on the normal calibration set and reused unchanged on every shifted set
    cal = {k: cq(y_ca, *pred(models, p_ca, k)) for k in ["M1", "M2", "M3"]}
    rows = []
    for name, p in sets.items():
        y = np.log10(robust_R(p)); _, F = feats(p, "M1"); r = dict(test_set=name, n=len(p), y_min=y.min(), y_max=y.max(), M0=rmse(F["logRq"].values, y))
        for k in ["M1", "M2", "M3"]:
            mu, sd = pred(models, p, k); r[k] = rmse(mu, y); r["cov_" + k] = float(np.mean(np.abs(y - mu) <= cal[k] * (sd + 1e-3)))
        rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/ood.csv", index=False)

# SEARCH: how well do different methods find a low-R pulse in a pool of candidates,
# for a given number B of exact evaluations?
elif PART == "SEARCH":
    p_tr, y_tr, _ = make_set(N_TRAIN, 101); models = fit_all(p_tr, y_tr)
    pool = gen_pulses(N_POOL, 999); Rtrue = np.log10(robust_R(pool))
    mu3, _ = pred(models, pool, "M3"); _, Fp = feats(pool, "M1"); m0 = Fp["logRq"].values
    nominal = Fp["logI0"].values
    budgets = [10, 20, 50, 100, 200, 500]; rows = []
    # simple evolution strategy: mutate the current best pulse, keep the best child, adapt the step size
    def es(B, seed, start, lam=10):
        r = np.random.default_rng(seed); x = start.copy(); fx = np.log10(robust_R(x[None]))[0]; used = 1; sc = 0.05
        while used + lam <= B:
            c = np.clip(x + sc * (HI - LO) * r.normal(size=(lam, 5)), LO, HI); yc = np.log10(robust_R(c)); used += lam; j = np.argmin(yc)
            if yc[j] < fx: x, fx = c[j], yc[j]; sc = min(sc * 1.2, 0.2)
            else: sc = max(sc * 0.7, 0.002)
        return fx
    rr = np.random.default_rng(3)
    for B in budgets:
        # random = best of B random pulses; es_cold / es_warm = evolution strategy from a random start / from the best nominal pulse
        rnd = [Rtrue[rr.choice(len(pool), B, replace=False)].min() for _ in range(30)]
        es_cold = [es(B, s, pool[rr.integers(len(pool))]) for s in range(10)]
        es_warm = [es(B, s, pool[np.argmin(nominal)]) for s in range(10)]
        # screening = rank the whole pool for free (nominal infidelity, physics M0, ML M3) and check only the top B exactly
        rows.append(dict(budget=B, random_mean=np.mean(rnd), random_std=np.std(rnd), ES_tuned_cold=np.mean(es_cold), ES_tuned_warm=np.mean(es_warm),
                         ES_warm_std=np.std(es_warm), M0_screen=Rtrue[np.argsort(m0)[:B]].min(), M3_screen=Rtrue[np.argsort(mu3)[:B]].min(),
                         nominal_screen=Rtrue[np.argsort(nominal)[:B]].min(), pool_best=Rtrue.min()))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/search_strong_baselines.csv", index=False)
    print("published:", np.log10(robust_R(PUB))[0])

# TIME: how long does the expensive Monte-Carlo label take compared with the ML prediction? (same run, same machine)
elif PART == "TIME":
    import run_extensions as rx
    p = gen_pulses(400, 7); K = 256; sig_e, sig_d = SIG_E, SIG_D
    rng = np.random.default_rng(77); half = rng.normal(size=(1, K // 2)) * sig_e; eps = np.concatenate([half, -half], axis=1)
    z = rx.ou_unit_paths(K // 2, 0.3, 11); z = np.concatenate([z, -z])
    # warm-up call so that start-up cost is not counted
    rx.R_mc(p[:20], eps, z, sig_d)
    # time per pulse of the Monte-Carlo label
    t = time.time(); rx.R_mc(p, eps, z, sig_d); t_mc = (time.time() - t) / len(p)
    p_tr = gen_pulses(800, 4001); X_tr, lq = rx.own_features(p_tr, sig_e, sig_d); m = Ens().fit(X_tr, np.log10(rx.R_mc(p_tr, eps, z, sig_d)) - lq)
    rx.own_features(p[:20], sig_e, sig_d)
    # time per pulse of the ML path (physics features + model prediction)
    t = time.time(); X, lqq = rx.own_features(p, sig_e, sig_d); mu, _ = m.predict(X); t_ml = (time.time() - t) / len(p)
    t = time.time(); robust_R(p); t_q = (time.time() - t) / len(p)
    res = dict(mc_ms_per_pulse=t_mc * 1e3, ml_ms_per_pulse_incl_features=t_ml * 1e3, quasistatic_exact_ms=t_q * 1e3, speedup=t_mc / t_ml)
    print(res); pd.DataFrame([res]).to_csv(f"{OUT}/timing_same_process.csv", index=False)
