# Rydberg CZ pulse robustness: simulator, features and pulse generator. Labels come from simulation only.
# Environment variables: PRESET, OUT, QUICK=1 (smoke test), SIG (both sigmas), SIGD (sigma_delta only).
import os, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from numpy.polynomial.hermite_e import hermegauss
from sklearn.ensemble import HistGradientBoostingRegressor

QUICK = os.environ.get("QUICK", "0") == "1"
OUT = os.environ.get("OUT", "results"); os.makedirs(OUT, exist_ok=True)

# the pulse is cut into NT small time-steps, and each step is treated as constant
NT      = 48
# GAMMA = decay rate of the Rydberg state, SIG_E = spread of the amplitude error, SIG_D = spread of the detuning error (units of Omega)
# 'stress' uses very large errors to see where the simple physics estimate breaks down
PRESETS = {"realistic": dict(GAMMA=3.9e-4, SIG_E=0.008, SIG_D=0.0163),
           "stress":    dict(GAMMA=3.9e-4, SIG_E=0.08,  SIG_D=0.08)}
_P = PRESETS[os.environ.get("PRESET", "realistic")]
GAMMA = _P["GAMMA"]
SIG_E = float(os.environ["SIG"]) if "SIG" in os.environ else _P["SIG_E"]
SIG_D = float(os.environ["SIG"]) if "SIG" in os.environ else float(os.environ.get("SIGD", _P["SIG_D"]))
# small step used to measure how the infidelity changes with the error (finite differences)
H_FD    = 0.03
N_TRAIN, N_CAL, N_TEST, N_POOL = (600, 300, 600, 4000) if QUICK else (3000, 1000, 2000, 30000)
HC_ITERS = 60 if QUICK else 200
ALPHA   = 0.10
# allowed range of the pulse parameters: A, omega, phi0, delta0, T
LO = np.array([0.0,        0.3, -np.pi, -1.5,  6.0])
HI = np.array([2*np.pi,    2.5,  np.pi,  1.5, 14.0])

# simulates one two-level block of the gate; returns the amplitude of ending back in the starting state
def _amp(p, eps, delta, c, gamma):
    """<g|U|g> for a batch.  p:(P,5), eps,delta:(P,K) -> (P,K) complex."""
    A, w, phi0, d0, T = [p[:, i][:, None] for i in range(5)]
    dt = T / NT
    # Rabi frequency of this block: c = 1 for |01>,|10> and sqrt(2) for |11>; eps is the amplitude error
    om = c * (1.0 + eps)
    hz = 0.5 * delta + 0.25j * gamma
    h0 = -0.5 * delta - 0.25j * gamma
    # exact 2x2 propagator for one time-step (the Hamiltonian is constant inside a step)
    kap = np.sqrt(0.25 * om**2 + hz**2 + 0j)
    x = kap * dt
    safe = np.where(np.abs(kap) > 1e-9, kap, 1.0)
    s = np.where(np.abs(kap) > 1e-9, np.sin(x) / safe, dt + 0 * kap)
    cc, ph = np.cos(x), np.exp(-1j * h0 * dt)
    a00, a11, b = ph * (cc - 1j * s * hz), ph * (cc + 1j * s * hz), ph * (-1j * s * 0.5 * om)
    v0 = np.ones(eps.shape, complex); v1 = np.zeros(eps.shape, complex)
    # apply the steps one after another; e is the laser phase factor in the middle of step k
    for k in range(NT):
        e = np.exp(-1j * (A * np.cos(w * (k + 0.5) * dt - phi0) + d0 * (k + 0.5) * dt))
        v0, v1 = a00 * v0 + b * e * v1, b * np.conj(e) * v0 + a11 * v1
    return v0

# CZ is only defined up to single-qubit phases, so find the phase theta that gives the best overlap with CZ
def _best_tr2(a, b):
    """max over single-qubit phase theta of |1 + 2 a z - b z^2|^2, z = e^{-i theta}."""
    best = np.full(a.shape, -1.0); bth = np.zeros(a.shape)
    for th in np.linspace(-np.pi, np.pi, 48, endpoint=False):
        z = np.exp(-1j * th); f = np.abs(1 + 2 * a * z - b * z * z) ** 2
        m = f > best; best = np.where(m, f, best); bth = np.where(m, th, bth)
    th = bth
    for _ in range(6):
        z = np.exp(-1j * th); T = 1 + 2 * a * z - b * z * z
        T1 = -2j * a * z + 2j * b * z * z; T2 = -2 * a * z + 4 * b * z * z
        f1 = 2 * np.real(np.conj(T) * T1); f2 = 2 * np.real(np.abs(T1) ** 2 + np.conj(T) * T2)
        th = th + np.clip(np.where(f2 < -1e-9, -f1 / f2, 0.0), -0.2, 0.2)
    z = np.exp(-1j * th)
    return np.abs(1 + 2 * a * z - b * z * z) ** 2

def infidelity(p, eps, delta, gamma=GAMMA):
    """1 - F_avg for CZ up to single-qubit Z phases (d = 4 average gate fidelity with leakage)."""
    a = _amp(p, eps, delta, 1.0, gamma); b = _amp(p, eps, delta, np.sqrt(2.0), gamma)
    # norm = total population left in the 4 basis states (decay and leakage reduce it)
    # fidelity = (norm + |overlap with CZ|^2) / 20, where 20 = d(d+1) for d = 4 states
    norm = 1 + 2 * np.abs(a) ** 2 + np.abs(b) ** 2
    F = (norm + _best_tr2(a, b)) / 20.0
    return np.clip(1.0 - F, 1e-12, None)

def _chunks(n, size=1000):
    for i in range(0, n, size): yield slice(i, min(i + size, n))

def nominal_I(p):
    out = np.empty(len(p))
    for sl in _chunks(len(p), 5000):
        z = np.zeros((sl.stop - sl.start, 1)); out[sl] = infidelity(p[sl], z, z)[:, 0]
    return out

# 7 points and weights for averaging over a Gaussian (Gauss-Hermite quadrature)
_x, _w = hermegauss(7); _w = _w / np.sqrt(2 * np.pi)

def robust_R(p, sig_e=SIG_E, sig_d=SIG_D):
    """Expected infidelity under Gaussian (eps, delta): deterministic 7x7 Gauss-Hermite quadrature."""
    E, D = np.meshgrid(sig_e * _x, sig_d * _x, indexing="ij"); W = np.outer(_w, _w).ravel()
    out = np.empty(len(p))
    for sl in _chunks(len(p), 500):
        n = sl.stop - sl.start
        out[sl] = (infidelity(p[sl], np.tile(E.ravel(), (n, 1)), np.tile(D.ravel(), (n, 1))) * W).sum(1)
    return out

# the published CZ pulse, used to check that the simulator is right
PUB = np.array([[2 * np.pi * 0.1122, 1.0431, -0.7318, 0.0, 2 * np.pi * 1.215]])
_z = np.zeros((1, 1))
I_noloss = infidelity(PUB, _z, _z, gamma=0.0)[0, 0]
I_decay = infidelity(PUB, _z, _z, gamma=1 / (88e-6 * 2 * np.pi * 4.6e6))[0, 0]
print(f"VALIDATION  published pulse, gamma=0: infidelity = {I_noloss:.2e} (should be ~1e-5 or below; NT={NT})")
print(f"VALIDATION  with T1 = 88 us decay: infidelity = {I_decay:.2e} (paper: 1.13e-3)")
assert I_noloss < 1e-4, "Simulator does not reproduce the published time-optimal CZ pulse - check before trusting anything below."

# physics features: 5 simulations per pulse (no error, and + or - a small error in amplitude and in detuning)
def phys_features(p):
    I = np.empty((len(p), 5))
    e = np.array([[0, H_FD, -H_FD, 0, 0]]); d = np.array([[0, 0, 0, H_FD, -H_FD]])
    for sl in _chunks(len(p), 4000):
        n = sl.stop - sl.start
        I[sl] = infidelity(p[sl], np.tile(e, (n, 1)), np.tile(d, (n, 1)))
    I0 = I[:, 0]
    see = np.clip((I[:, 1] + I[:, 2] - 2 * I0) / (2 * H_FD**2), 1e-9, None)
    sdd = np.clip((I[:, 3] + I[:, 4] - 2 * I0) / (2 * H_FD**2), 1e-9, None)
    # physics-only estimate of R: infidelity without errors + curvature x (error size)^2
    Rq = I0 + see * SIG_E**2 + sdd * SIG_D**2
    return pd.DataFrame({"logI0": np.log10(I0), "logSee": np.log10(see),
                         "logSdd": np.log10(sdd), "logRq": np.log10(Rq)})

def raw_features(p):
    return pd.DataFrame({"A": p[:, 0], "w": p[:, 1], "cphi0": np.cos(p[:, 2]), "sphi0": np.sin(p[:, 2]),
                         "d0": p[:, 3], "T": p[:, 4]})

# makes pulses by random hill-climbing on the infidelity without errors, so they are reasonably good gates
def gen_pulses(n, seed, iters=HC_ITERS):
    rng = np.random.default_rng(seed)
    p = rng.uniform(LO, HI, (n, 5)); f = nominal_I(p); snaps = []
    for it in range(iters):
        sc = 0.15 * (1 - it / iters) + 0.005
        q = np.clip(p + sc * (HI - LO) * rng.normal(size=p.shape) * (rng.random(p.shape) < 0.6), LO, HI)
        fq = nominal_I(q); m = fq < f; p[m] = q[m]; f[m] = fq[m]
        if it == iters // 3: snaps.append(p.copy())
    snaps.append(p.copy())
    return np.vstack(snaps)[rng.permutation(len(snaps) * n)][:n]

# returns pulses, their exact log10 R values (the labels) and the time per pulse
def make_set(n, seed):
    t = time.time(); p = gen_pulses(n, seed); t1 = time.time() - t
    t = time.time(); R = robust_R(p); t2 = time.time() - t
    return p, np.log10(R), t2 / len(p)
