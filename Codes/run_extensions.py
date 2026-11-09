# Extensions: A) time-correlated detuning noise, B) finite Rydberg blockade. Needs rydberg_core.py in the same folder.
import rydberg_core as rc
from rydberg_core import *
from sklearn.ensemble import HistGradientBoostingRegressor

EXT = os.path.join(OUT, "extensions"); os.makedirs(EXT, exist_ok=True)
# time grid (units of 1/Omega) on which the noise paths are generated
DT_F = 0.02

class Ens:
    def __init__(self, k=5, seed=0): self.k, self.seed, self.m = k, seed, []
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed); self.m = []
        for i in range(self.k):
            idx = rng.integers(0, len(X), len(X))
            self.m.append(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                          l2_regularization=1e-3, random_state=i).fit(X.iloc[idx], y[idx]))
        return self
    def predict(self, X): return np.mean([m.predict(X) for m in self.m], axis=0)

# noise with memory (Ornstein-Uhlenbeck): tau = how long the noise stays correlated; tau = inf means constant during the pulse
def ou_unit_paths(K, tau, seed, t_max=15.0):
    """K stationary unit-variance OU paths, correlation time tau (units 1/Omega), on grid DT_F. tau=inf -> frozen."""
    rng = np.random.default_rng(seed); n = int(t_max / DT_F) + 2
    z = np.empty((K, n)); z[:, 0] = rng.normal(size=K); xi = rng.normal(size=(K, n))
    if np.isinf(tau): return np.repeat(z[:, :1], n, axis=1)
    a = np.exp(-DT_F / tau); s = np.sqrt(1 - a * a)
    # each value keeps a fraction a of the previous one and adds fresh random noise
    for i in range(1, n): z[:, i] = a * z[:, i - 1] + s * xi[:, i]
    return z

# same idea as _amp in rydberg_core.py, but the detuning changes from step to step
def _amp_t(p, eps, zz, c, gamma):
    """<g|U|g> with time-dependent detuning zz:(P,K,NT) (already scaled to Omega units); eps:(1,K)."""
    A, w, phi0, d0, T = [p[:, i][:, None] for i in range(5)]
    dt = T / NT; P, K = len(p), eps.shape[1]
    om = c * (1.0 + eps)
    v0 = np.ones((P, K), complex); v1 = np.zeros((P, K), complex)
    for k in range(NT):
        d = zz[:, :, k]
        hz = 0.5 * d + 0.25j * gamma; h0 = -0.5 * d - 0.25j * gamma
        kap = np.sqrt(0.25 * om**2 + hz**2 + 0j); x = kap * dt
        s = np.sin(x) / kap; cc = np.cos(x); ph = np.exp(-1j * h0 * dt)
        a00, a11, b = ph * (cc - 1j * s * hz), ph * (cc + 1j * s * hz), ph * (-1j * s * 0.5 * om)
        t = (k + 0.5) * dt
        e = np.exp(-1j * (A * np.cos(w * t - phi0) + d0 * t))
        v0, v1 = a00 * v0 + b * e * v1, b * np.conj(e) * v0 + a11 * v1
    return v0

# Monte-Carlo estimate of R: average the infidelity over many noise samples (the same samples for every pulse)
def R_mc(p, eps, z, sig_d, gamma=GAMMA):
    """Monte-Carlo expected infidelity; SAME noise realisations for every pulse (common random numbers)."""
    out = np.empty(len(p))
    for sl in rc._chunks(len(p), 200):
        q = p[sl]; dt = q[:, 4:5] / NT
        tk = (np.arange(NT)[None, :] + 0.5) * dt
        idx = np.clip(np.rint(tk / DT_F).astype(int), 0, z.shape[1] - 1)
        zz = np.transpose(sig_d * z[:, idx], (1, 0, 2))
        a = _amp_t(q, eps, zz, 1.0, gamma); b = _amp_t(q, eps, zz, np.sqrt(2.0), gamma)
        F = (1 + 2 * np.abs(a)**2 + np.abs(b)**2 + rc._best_tr2(a, b)) / 20.0
        out[sl] = np.clip(1 - F, 1e-12, None).mean(1)
    return out

# physics features for any noise size (used when the noise is scaled up)
def own_features(p, sig_e, sig_d):
    X, F = raw_features(p), phys_features(p)
    I0, see, sdd = 10 ** F["logI0"].values, 10 ** F["logSee"].values, 10 ** F["logSdd"].values
    logRq = np.log10(I0 + see * sig_e**2 + sdd * sig_d**2)
    return pd.concat([X, F[["logI0", "logSee", "logSdd"]].assign(logRq=logRq)], axis=1), logRq

# part A: does the quasi-static physics estimate still work when the noise is correlated in time?
def part_A():
    K = 64 if QUICK else 256; ntr, nte = (300, 150) if QUICK else (800, 400)
    taus = [0.3, 3.0, np.inf] if QUICK else [0.1, 0.3, 1.0, 3.0, 10.0, np.inf]
    scales = [1, 5]
    p_tr = gen_pulses(ntr, 4001); p_te = gen_pulses(nte, 4002); rows = []
    for sc in scales:
        sig_e, sig_d = SIG_E * sc, SIG_D * sc
        rng = np.random.default_rng(77)
        half = rng.normal(size=(1, K // 2)) * sig_e; eps = np.concatenate([half, -half], axis=1)
        X_tr, lq_tr = own_features(p_tr, sig_e, sig_d); X_te, lq_te = own_features(p_te, sig_e, sig_d)
        zs = ou_unit_paths(K // 2, np.inf, 11); zs = np.concatenate([zs, -zs])
        y_st_te = np.log10(R_mc(p_te, eps, zs, sig_d))
        floor = np.sqrt(np.mean((lq_te - y_st_te) ** 2))
        # for each correlation time: exact Monte-Carlo labels, then fit M2 (ML + physics features) and M3 (physics + ML correction)
        for tau in taus:
            z = ou_unit_paths(K // 2, tau, 11); z = np.concatenate([z, -z])
            t0 = time.time(); y_tr = np.log10(R_mc(p_tr, eps, z, sig_d)); t_mc = (time.time() - t0) / len(p_tr)
            y_te = np.log10(R_mc(p_te, eps, z, sig_d))
            m2 = Ens().fit(X_tr, y_tr).predict(X_te)
            m3 = Ens().fit(X_tr, y_tr - lq_tr).predict(X_te) + lq_te
            r = lambda m: float(np.sqrt(np.mean((m - y_te) ** 2)))
            rows.append(dict(noise_scale=sc, sigma_eps=sig_e, sigma_delta=sig_d, tau_c=tau, N_mc=K,
                             RMSE_M0_quasistatic_physics=r(lq_te), RMSE_M2_ML_physics=r(m2), RMSE_M3_physics_plus_ML=r(m3),
                             sampling_floor_M0_vs_static_MC=floor,
                             median_ratio_R_over_Rstatic=float(np.median(10 ** (y_te - y_st_te))), mc_ms_per_pulse=t_mc * 1e3))
            print(rows[-1])
    df = pd.DataFrame(rows); df.to_csv(f"{EXT}/A_correlated_noise.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for sc, c in zip(scales, ["C0", "C3"]):
        d = df[df.noise_scale == sc].copy(); d["x"] = d.tau_c.replace(np.inf, 100.0)
        ax[0].plot(d.x, d.RMSE_M0_quasistatic_physics, "o-", c=c, label=f"M0 physics, noise x{sc}")
        ax[0].plot(d.x, d.RMSE_M3_physics_plus_ML, "s--", c=c, label=f"M3 physics+ML, noise x{sc}")
        ax[1].plot(d.x, d.median_ratio_R_over_Rstatic, "o-", c=c, label=f"noise x{sc}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("correlation time tau_c (1/Omega); right edge = quasi-static"); ax[0].set_ylabel("RMSE of log10 R")
    ax[0].set_title("Prediction error vs noise correlation time"); ax[0].legend(fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("tau_c (1/Omega)"); ax[1].set_ylabel("median R(tau_c) / R(quasi-static)"); ax[1].set_title("Benefit of fast noise"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{EXT}/fig_A_correlated_noise.png", dpi=150); plt.show()

# part B: 3-level block (|11>, |W>, |rr>) with a finite blockade strength B
def b_amp_finiteB(p, B, eps, delta, gamma):
    """<11|U|11> in the 3-level block (|11>,|W>,|rr>) with blockade shift B (units of Omega); Strang-split decay."""
    A, w, phi0, d0, T = [p[:, i][:, None] for i in range(5)]
    dt = T / NT; g = (1 + eps) / np.sqrt(2.0) + 0 * delta; Pn, K = g.shape
    psi = np.zeros((Pn, K, 3), complex); psi[..., 0] = 1
    damp = np.exp(-np.array([0.0, gamma / 2, gamma]) * 0.5 * dt[..., None])
    for k in range(NT):
        t = (k + 0.5) * dt; e = np.exp(-1j * (A * np.cos(w * t - phi0) + d0 * t))
        H = np.zeros((Pn, K, 3, 3), complex)
        H[..., 0, 1] = g * e; H[..., 1, 0] = g * np.conj(e); H[..., 1, 2] = g * e; H[..., 2, 1] = g * np.conj(e)
        H[..., 1, 1] = -delta; H[..., 2, 2] = B - 2 * delta
        E, V = np.linalg.eigh(H)
        psi = psi * damp
        c = np.einsum("pkji,pkj->pki", V.conj(), psi) * np.exp(-1j * E * dt[..., None])
        psi = np.einsum("pkij,pkj->pki", V, c) * damp
    return psi[..., 0]

def infid_B(p, B, eps, delta, gamma=GAMMA):
    a = rc._amp(p, eps, delta, 1.0, gamma)
    b = rc._amp(p, eps, delta, np.sqrt(2.0), gamma) if np.isinf(B) else b_amp_finiteB(p, B, eps, delta, gamma)
    return np.clip(1 - (1 + 2 * np.abs(a)**2 + np.abs(b)**2 + rc._best_tr2(a, b)) / 20.0, 1e-12, None)

def R_B(p, B):
    E, D = np.meshgrid(SIG_E * rc._x, SIG_D * rc._x, indexing="ij"); W = np.outer(rc._w, rc._w).ravel(); out = np.empty(len(p))
    for sl in rc._chunks(len(p), 100):
        n = sl.stop - sl.start
        out[sl] = (infid_B(p[sl], B, np.tile(E.ravel(), (n, 1)), np.tile(D.ravel(), (n, 1))) * W).sum(1)
    return out

# part B: do pulses that are robust at infinite blockade stay robust when B is finite?
def part_B():
    z = np.zeros((1, 1)); pub = PUB.copy()
    chk = infid_B(pub, 1e5, z, z)[0, 0], infid_B(pub, np.inf, z, z)[0, 0]
    print(f"CHECK finite-B code at B=1e5 Omega vs infinite blockade (published pulse): {chk[0]:.3e} vs {chk[1]:.3e}")
    pool = gen_pulses(600 if QUICK else 3000, 5001); Rinf = R_B(pool, np.inf)
    cand = pool[np.argsort(Rinf)[: (60 if QUICK else 300)]]; cand = np.vstack([cand, pub])
    # blockade strengths B / Omega to test (inf = perfect blockade)
    Bs = [np.inf, 100.0, 30.0, 16.0, 8.0]; res = {}; rows = []
    for B in Bs:
        res[B] = np.log10(R_B(cand, B))
    best_inf = int(np.argmin(res[np.inf][:-1]))
    for B in Bs:
        y = res[B]; best_B = int(np.argmin(y[:-1]))
        rows.append(dict(B_over_Omega=B, median_log10R=float(np.median(y[:-1])), published_log10R=float(y[-1]),
                         spearman_vs_infinite=float(spearmanr(y[:-1], res[np.inf][:-1])[0]),
                         log10R_of_infiniteB_best=float(y[best_inf]), log10R_of_best_at_this_B=float(y[best_B]),
                         regret_log10=float(y[best_inf] - y[best_B])))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(f"{EXT}/B_finite_blockade.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(res[np.inf][:-1], res[16.0][:-1], s=8); lim = [min(res[np.inf].min(), res[16.0].min()), max(res[np.inf].max(), res[16.0].max())]
    ax[0].plot(lim, lim, "k--"); ax[0].set_xlabel("log10 R, infinite blockade"); ax[0].set_ylabel("log10 R, B = 16 Omega"); ax[0].set_title("Ranking transfer")
    d = pd.DataFrame(rows); ax[1].plot(np.where(np.isinf(d.B_over_Omega), 300, d.B_over_Omega), d.regret_log10, "o-")
    ax[1].set_xscale("log"); ax[1].set_xlabel("B / Omega (right edge = infinite)"); ax[1].set_ylabel("regret in log10 R"); ax[1].set_title("Cost of designing at infinite blockade")
    plt.tight_layout(); plt.savefig(f"{EXT}/fig_B_finite_blockade.png", dpi=150); plt.show()

# PART=A, B or AB chooses what to run
if __name__ == "__main__":
    which = os.environ.get("PART", "AB")
    t0 = time.time()
    if "B" in which: part_B()
    if "A" in which: part_A()
    print(f"extensions done in {time.time()-t0:.0f}s -> {EXT}")
