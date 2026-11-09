# Main pipeline. Needs rydberg_core.py in the same folder.
import rydberg_core as rc
from rydberg_core import *

t0 = time.time()
# three independent datasets (different seeds): training, calibration (used for the intervals) and test
p_tr, y_tr, t_exact = make_set(N_TRAIN, 101)
p_ca, y_ca, _ = make_set(N_CAL, 202)
p_te, y_te, _ = make_set(N_TEST, 303)
print(f"datasets built in {time.time()-t0:.0f}s; exact robust_R cost = {t_exact*1e3:.2f} ms/pulse")
pd.DataFrame(np.c_[p_tr, y_tr], columns=["A", "omega", "phi0", "delta0", "T", "log10R"]).to_csv(f"{OUT}/train.csv", index=False)
pd.DataFrame(np.c_[p_te, y_te], columns=["A", "omega", "phi0", "delta0", "T", "log10R"]).to_csv(f"{OUT}/test.csv", index=False)

# 5 gradient-boosted tree models trained on resampled data: their average is the prediction, their spread is the uncertainty
class Ens:
    """Bootstrap ensemble of gradient-boosted trees -> mean and std (std feeds normalised conformal)."""
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

# M1 uses only the raw pulse parameters; M2 and M3 also get the physics features
def feats(p, kind):
    X, F = raw_features(p), phys_features(p)
    return {"M1": X, "M2": pd.concat([X, F], axis=1), "M3": pd.concat([X, F], axis=1)}[kind], F

# M1 and M2 learn log10 R directly; M3 learns only the correction to the physics estimate
def fit_all(p, y):
    models = {}
    for k in ["M1", "M2", "M3"]:
        X, F = feats(p, k); tgt = y - F["logRq"].values if k == "M3" else y
        models[k] = Ens().fit(X, tgt)
    return models

def predict(models, p, k):
    X, F = feats(p, k); mu, sd = models[k].predict(X)
    return (mu + F["logRq"].values if k == "M3" else mu), sd

# conformal prediction: the error size (in units of the spread) that covers 90% of the calibration pulses
def conformal_q(y, mu, sd, alpha=ALPHA):
    s = np.abs(y - mu) / (sd + 1e-3); n = len(s)
    return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))

def top_recall(y, yhat, true_frac=0.01, pred_frac=0.05):
    kt, kp = max(1, int(true_frac * len(y))), max(1, int(pred_frac * len(y)))
    return len(set(np.argsort(y)[:kt]) & set(np.argsort(yhat)[:kp])) / kt

# train M1, M2, M3 and score them on the test set, together with M0 (physics only)
models = fit_all(p_tr, y_tr)
rows, preds = [], {}
_, F_te = feats(p_te, "M1")
preds["M0 perturbative"] = (F_te["logRq"].values, None)
for k, name in [("M1", "M1 ML raw"), ("M2", "M2 ML raw+physics"), ("M3", "M3 perturbative+ML residual")]:
    preds[name] = predict(models, p_te, k)
for name, (mu, sd) in preds.items():
    row = dict(model=name, RMSE_log10=np.sqrt(np.mean((mu - y_te) ** 2)),
               MAE_log10=np.mean(np.abs(mu - y_te)), Spearman=spearmanr(mu, y_te)[0],
               top1pct_in_pred5pct=top_recall(y_te, mu))
    if sd is not None:
        k = {"M1 ML raw": "M1", "M2 ML raw+physics": "M2", "M3 perturbative+ML residual": "M3"}[name]
        # find q on the calibration set, then check how many test pulses fall inside the interval (should be close to 0.9)
        mc, sc = predict(models, p_ca, k); q = conformal_q(y_ca, mc, sc)
        row["coverage_90"] = np.mean(np.abs(y_te - mu) <= q * (sd + 1e-3))
    rows.append(row)
metrics = pd.DataFrame(rows); print(metrics.round(4).to_string(index=False))
metrics.to_csv(f"{OUT}/metrics_in_distribution.csv", index=False)

# extrapolation test: train on short pulses (T < 9.5) and test on long ones (T >= 9.5)
tr_m, te_m = p_tr[:, 4] < 9.5, p_te[:, 4] >= 9.5
if tr_m.sum() > 100 and te_m.sum() > 50:
    m_sub = fit_all(p_tr[tr_m], y_tr[tr_m]); ood = []
    _, Fo = feats(p_te[te_m], "M1")
    ood.append(dict(model="M0 perturbative", RMSE_log10=np.sqrt(np.mean((Fo["logRq"].values - y_te[te_m]) ** 2))))
    for k in ["M1", "M2", "M3"]:
        mu, _ = predict(m_sub, p_te[te_m], k)
        ood.append(dict(model=k, RMSE_log10=np.sqrt(np.mean((mu - y_te[te_m]) ** 2)), Spearman=spearmanr(mu, y_te[te_m])[0]))
    pd.DataFrame(ood).to_csv(f"{OUT}/metrics_extrapolation.csv", index=False); print(pd.DataFrame(ood).round(4))

# time the ML prediction and compare it with the exact calculation
t = time.time(); _ = predict(models, p_te[:1000], "M3"); t_ml = (time.time() - t) / 1000
print(f"exact R: {t_exact*1e3:.3f} ms/pulse | ML(M3, incl. 5-sim features): {t_ml*1e3:.3f} ms/pulse | ratio {t_exact/t_ml:.1f}x")

# search test: rank a pool of candidate pulses with M3, then check only the top B with the exact simulator
pool = gen_pulses(N_POOL, 999)
mu_pool, sd_pool = predict(models, pool, "M3")
R_pool_true = np.log10(robust_R(pool))
budgets = [20, 50, 100, 200, 500] if QUICK else [20, 50, 100, 200, 500, 1000]
rng = np.random.default_rng(7); order_ml = np.argsort(mu_pool); res = []
def es_direct(B, seed):
    """(1+lambda)-ES directly on exact R, lambda = 20."""
    r = np.random.default_rng(seed); lam = 20
    best = pool[r.integers(len(pool), size=lam)]; yb = np.log10(robust_R(best)); i = np.argmin(yb)
    x, fx, used, hist = best[i], yb[i], lam, []
    while used + lam <= B:
        sc = 0.08 * (1 - used / B) + 0.005
        c = np.clip(x + sc * (HI - LO) * r.normal(size=(lam, 5)), LO, HI); yc = np.log10(robust_R(c)); used += lam
        j = np.argmin(yc)
        if yc[j] < fx: x, fx = c[j], yc[j]
    return fx
# for each budget B: best pulse found by random choice, by ML ranking and by direct search
for B in budgets:
    rnd = np.mean([R_pool_true[rng.choice(len(pool), B, replace=False)].min() for _ in range(20)])
    res.append(dict(budget=B, random=rnd, ML_screen=R_pool_true[order_ml[:B]].min(),
                    direct_ES=np.mean([es_direct(B, s) for s in range(3)]), pool_oracle_best=R_pool_true.min()))
res = pd.DataFrame(res); print(res.round(3).to_string(index=False)); res.to_csv(f"{OUT}/search_comparison_log10R.csv", index=False)

R_pub = np.log10(robust_R(PUB))[0]
print(f"Published Evered pulse: log10 R = {R_pub:.3f}; fraction of pool pulses with lower R = {(R_pool_true < R_pub).mean():.3f}; pool best = {R_pool_true.min():.3f}")
pd.DataFrame([dict(published_log10R=R_pub, pool_best_log10R=R_pool_true.min(), frac_pool_better=(R_pool_true < R_pub).mean())]).to_csv(f"{OUT}/published_pulse_baseline.csv", index=False)

# best pulse among the 200 top-ranked by M3
best = pool[order_ml[:200]][np.argmin(R_pool_true[order_ml[:200]])]
np.save(f"{OUT}/best_pulse_params.npy", best)

# figures: predicted vs exact, error of each model, and the search results
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
mu, sd = preds["M3 perturbative+ML residual"]
ax[0].scatter(y_te, mu, s=4, alpha=.5); ax[0].plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "k--")
ax[0].set_xlabel("exact log10 R"); ax[0].set_ylabel("predicted log10 R (M3)"); ax[0].set_title("Held-out parity")
ax[1].bar(metrics["model"].str.split().str[0], metrics["RMSE_log10"]); ax[1].set_ylabel("RMSE of log10 R"); ax[1].set_title("Test error by model")
for c, lab in [("random", "random"), ("direct_ES", "direct ES"), ("ML_screen", "ML screen")]:
    ax[2].plot(res["budget"], res[c], "o-", label=lab)
ax[2].axhline(res["pool_oracle_best"].iloc[0], color="k", ls=":", label="pool oracle"); ax[2].set_xscale("log")
ax[2].set_xlabel("exact R evaluations"); ax[2].set_ylabel("best log10 R found"); ax[2].legend(); ax[2].set_title("Robust-pulse search")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_main.png", dpi=150); plt.show()

e = np.linspace(-.1, .1, 41); d = np.linspace(-.1, .1, 41); E, D = np.meshgrid(e, d)
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); idx_nom = np.argmin(nominal_I(pool))
for a_, pp, ttl in [(ax[0], pool[idx_nom], "best NOMINAL pulse"), (ax[1], best, "ML-screened robust pulse")]:
    Z = infidelity(np.tile(pp, (1, 1)), E.reshape(1, -1), D.reshape(1, -1)).reshape(E.shape)
    im = a_.contourf(E, D, np.log10(Z), 20); plt.colorbar(im, ax=a_, label="log10 infidelity")
    a_.set_xlabel("amplitude error"); a_.set_ylabel("detuning error / Omega"); a_.set_title(ttl)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_landscape.png", dpi=150); plt.show()
json.dump(dict(PRESET=os.environ.get("PRESET", "realistic"), QUICK=QUICK, GAMMA=GAMMA, SIG_E=SIG_E, SIG_D=SIG_D, N_TRAIN=N_TRAIN, N_CAL=N_CAL, N_TEST=N_TEST, N_POOL=N_POOL),
          open(f"{OUT}/config.json", "w"))
