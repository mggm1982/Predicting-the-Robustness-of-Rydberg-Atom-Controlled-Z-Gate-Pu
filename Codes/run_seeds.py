# Repeats train/test generation and fitting with independent seeds. Usage: PRESET=realistic|stress [SIG=0.15] [NSEEDS=5] OUT=<folder> python run_seeds.py
import rydberg_core as rc
from rydberg_core import *
# same models as in run_pipeline.py, but returning only the predicted values (no uncertainty)
class Ens:
    def __init__(self, k=5, seed=0): self.k, self.seed, self.m = k, seed, []
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed); self.m = []
        for i in range(self.k):
            idx = rng.integers(0, len(X), len(X))
            self.m.append(HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                          l2_regularization=1e-3, random_state=i).fit(X.iloc[idx], y[idx]))
        return self
    def predict(self, X): return np.array([m.predict(X) for m in self.m]).mean(0)
def feats(p, kind):
    X, F = raw_features(p), phys_features(p)
    return {"M1": X, "M2": pd.concat([X, F], axis=1), "M3": pd.concat([X, F], axis=1)}[kind], F
def fit_all(p, y):
    out = {}
    for k in ["M1", "M2", "M3"]:
        X, F = feats(p, k); out[k] = Ens().fit(X, y - F["logRq"].values if k == "M3" else y)
    return out
def pred(models, p, k):
    X, F = feats(p, k); mu = models[k].predict(X); return mu + F["logRq"].values if k == "M3" else mu
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
rows = []
# repeat the whole experiment with independent random datasets to see how much the results change
for s in range(int(os.environ.get("NSEEDS", 5))):
    p_tr, y_tr, _ = make_set(N_TRAIN, 1000 + s); p_te, y_te, _ = make_set(N_TEST, 2000 + s)
    m = fit_all(p_tr, y_tr); _, F = feats(p_te, "M1")
    r = dict(seed=s, M0=rmse(F["logRq"].values, y_te), **{k: rmse(pred(m, p_te, k), y_te) for k in ["M1", "M2", "M3"]})
    # extrapolation: train on T < 9.5, test on T >= 9.5
    a, b = p_tr[:, 4] < 9.5, p_te[:, 4] >= 9.5
    ms = fit_all(p_tr[a], y_tr[a]); _, Fo = feats(p_te[b], "M1")
    r.update(xM0=rmse(Fo["logRq"].values, y_te[b]), **{"x" + k: rmse(pred(ms, p_te[b], k), y_te[b]) for k in ["M1", "M2", "M3"]})
    rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/seeds.csv", index=False)
print(df.drop(columns="seed").agg(["mean", "std"]).round(4).to_string())
