"""Sec. IV-B: can a cascade of rings synthesize the softmax exponential?
Two configurations are compared over x in [-3,0] (softmax logits after subtracting the max):
  (1) EXTERNALLY DETUNED cascade -- each stage a Lorentzian in the logit, stages identical:
      T_N(x) = [1 + ((x-x0)/w)^2]^{-N}
  (2) SELF-ACTION cascade -- solved self-consistently from the paper's own cubic.
      With P_drop = gamma_c2 * U, one stage maps input power x to the root of
      g(y) = y[(Delta-y)^2 + 1] = x, so a chain is the composition g^{-N}
      (each stage driven by the attenuated output of the last; stages are NOT identical).
Result: (1) reaches 1.8% at N=3 and keeps improving; (2) plateaus at 14-17% and does not.
"""
import numpy as np
from scipy.optimize import brentq, minimize

DELTA = 1.6                      # below the sqrt(3) bistability threshold, as in Sec. VI-A
xt = np.linspace(-3.0, 0.0, 200); tgt = np.exp(xt)

# ---- (1) externally detuned cascade ----
def err_ext(p, N):
    x0, w = p
    if w <= 1e-3: return 1e3
    T = 1.0/(1.0 + ((xt-x0)/w)**2)
    g = (T**N)/((1.0/(1.0+(x0/w)**2))**N)
    return np.max(np.abs(g-tgt)/tgt)

# ---- (2) self-action cascade ----
def g_fwd(y):  return y*((DELTA-y)**2 + 1.0)
def g_inv(x):  return brentq(lambda y: g_fwd(y)-x, 0.0, 50.0, xtol=1e-14)
def cascade(x, N):
    for _ in range(N): x = g_inv(x)
    return x
def err_self(p, N):
    s, off, A = p
    if s <= 1e-3 or A <= 0: return 1e3
    xi = off + s*(xt + 3.0)
    if xi.min() <= 1e-4: return 1e3
    return np.max(np.abs(A*np.array([cascade(v, N) for v in xi]) - tgt)/tgt)

def best(fn, N, seeds):
    return min((minimize(fn, p0, args=(N,), method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-8, fatol=1e-12)) for p0 in seeds),
               key=lambda z: z.fun)

print(" N | externally detuned | insertion loss | self-action")
for N in (1, 2, 3, 4, 5):
    re_ = best(err_ext,  N, ([2.,2.], [1.,1.5], [4.,3.]))
    rs_ = best(err_self, N, ([1.,.2,1.], [.5,.05,2.], [2.,.5,.5]))
    x0, w = re_.x; T0 = (1.0/(1.0+(x0/w)**2))**N
    print(f"{N:2d} | {re_.fun*100:17.1f}% | {-10*np.log10(T0):11.1f} dB | {rs_.fun*100:9.1f}%")
print(f"\nself-action small-signal attenuation per stage: "
      f"{-10*np.log10(1/(DELTA**2+1)):.1f} dB  (= 1/(Delta^2+1))")

# ---- sweep Delta over the whole single-valued regime (Delta < sqrt3) ----
# g'(y) = 3u^2 - 2*Delta*u + 1 with u = Delta - y has real roots only for Delta >= sqrt3,
# so g is invertible -- and g^{-N} defined -- exactly where Remark 1 demands single-valued
# operation. Sweeping inside that regime asks whether some other detuning suits the
# exponential better than the activation's own Delta = 1.6.
def inv_interp(D):
    y = np.linspace(0, 60, 300000); x = y*((D-y)**2 + 1.0)
    return lambda xs: np.interp(xs, x, y)
def err_self_D(p, N, ginv):
    s_, off, A = p
    if s_ <= 1e-4 or A <= 0: return 1e3
    xi = off + s_*(xt + 3.0)
    if xi.min() <= 1e-5: return 1e3
    z = xi.copy()
    for _ in range(N): z = ginv(z)
    return np.max(np.abs(A*z - tgt)/tgt)
print("\nDelta sweep (self-action, worst-case % error):")
print("  Delta |   N=1    N=2    N=3    N=4")
bestD = (1e9, None)
for D in (0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.7):
    gi = inv_interp(D); row = []
    for N in (1, 2, 3, 4):
        r = min((minimize(err_self_D, p0, args=(N, gi), method="Nelder-Mead",
                          options=dict(maxiter=1200, xatol=1e-6, fatol=1e-10))
                 for p0 in ([1., .2, 1.], [.4, .05, 2.])), key=lambda z: z.fun)
        row.append(r.fun*100)
        if r.fun < bestD[0]: bestD = (r.fun, (D, N))
    print(f"  {D:5.1f} | " + " ".join(f"{v:6.1f}" for v in row))
print(f"  best over the single-valued regime: {bestD[0]*100:.1f}% at Delta={bestD[1][0]}, N={bestD[1][1]}"
      f"  ({-10*np.log10((1/(bestD[1][0]**2+1))**bestD[1][1]):.1f} dB total)")
