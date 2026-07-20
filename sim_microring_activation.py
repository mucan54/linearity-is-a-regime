"""Microring self-action activation (Sec. 6.1, Fig. fig_activation).
Solves the normalized Kerr coupled-mode model just below the bistability
threshold and fits the drop-port transfer to sigmoid/GELU/SiLU.
Writes figures/fig_activation.{pdf,png} and figures/act_{xn,yn}.npy
(the latter are consumed by sim_singleblock_noise_probe.py)."""
import os; os.makedirs("figures",exist_ok=True)

import numpy as np
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, json

# ---------- Canonical normalized Kerr microring (CMT) ----------
# Normalized steady state (loss rate -> 1):  x_in = y*[(Delta - y)^2 + 1]
#   y      : normalized circulating power (|a|^2)
#   Delta  : normalized cold-cavity detuning (blue side positive)
#   x_in   : normalized input power
# Bistability threshold at Delta = sqrt(3) ~ 1.732. Operate just below -> strong monostable nonlinearity.
# Drop-port transmission (add-drop):  T(y) = 1/((Delta - y)^2 + 1)  (normalized peak=1)
Delta = 1.60   # just below sqrt(3): strong, single-valued nonlinearity

def circulating(xin, Delta):
    # invert x_in = y*((Delta-y)^2+1) for smallest positive real root (lower/physical branch)
    # cubic in y: y^3 -2*Delta*y^2 + (Delta^2+1)*y - xin = 0
    coeffs=[1.0, -2*Delta, (Delta**2+1), -xin]
    roots=np.roots(coeffs)
    real=[r.real for r in roots if abs(r.imag)<1e-9 and r.real>=-1e-9]
    return min(real) if real else np.nan

xin = np.linspace(1e-4, 4.5, 500)
y   = np.array([circulating(x, Delta) for x in xin])
T   = 1.0/((Delta - y)**2 + 1.0)
xout= xin * T                      # output power (normalized)

# normalize to [0,1] for activation comparison
xn = xin/xin.max()
yn = xout/xout.max()

def gelu(x): return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def silu(x): return x/(1+np.exp(-x))
def softplus(x): return np.log1p(np.exp(x))
def relu(x): return np.maximum(0,x)
def sigmoid(x): return 1/(1+np.exp(-x))

def fit_target(fn):
    def model(x,a,b,d,e): return a*fn(b*x+d)+e
    best=None
    for p0 in ([1,4,-2,0],[1,6,-3,0],[1,3,-1,0],[0.8,5,-2.5,0.1]):
        try:
            popt,_=curve_fit(model,xn,yn,p0=p0,maxfev=40000)
            yp=model(xn,*popt); ssr=np.sum((yn-yp)**2); sst=np.sum((yn-yn.mean())**2)
            r2=1-ssr/sst
            if best is None or r2>best[0]: best=(r2,yp,popt)
        except Exception: pass
    return best if best else (-9,None,None)

res={}
for name,fn in [("GELU",gelu),("SiLU",silu),("Softplus",softplus),("Sigmoid",sigmoid),("ReLU",relu)]:
    r2,yp,popt=fit_target(fn); res[name]=(r2,yp)
    print(f"{name:9s} R^2={r2:.4f}")
best=max(res,key=lambda k:res[k][0])
print("BEST:",best,"R^2=",round(res[best][0],4))

# physical anchoring: characteristic Kerr power P_c for a realistic high-Q ring
lam0=1550e-9; c=3e8; w0=2*np.pi*c/lam0
Q=5e5; n0=2.4; n2=4.5e-18; Veff=3e-18
# characteristic (bistability) input power scale ~ n0^2 Veff w0 /(4 c n2 Q^2)  (order-of-mag)
Pc = (n0**2 * Veff * w0)/(4*c*n2*Q**2)
print(f"Characteristic Kerr power P_c ~ {Pc*1e3:.3f} mW  (Q={Q:.0e})")
print(f"Operating input range: {xin.max()*Pc*1e3:.2f} mW  (x_in_max={xin.max():.1f} * P_c)")

fig,ax=plt.subplots(1,2,figsize=(11,4.2))
ax[0].plot(xin, xout, lw=2.3, color="#1b3a6b")
ax[0].set_xlabel(r"Normalized input $x_{in}$ (units of $P_c$)")
ax[0].set_ylabel(r"Normalized drop output")
ax[0].set_title(f"Kerr microring transfer ($\\Delta$={Delta}, below bistability)")
ax[0].grid(alpha=0.3)
ax[1].plot(xn,yn,lw=2.5,color="#1b3a6b",label="Microring (simulated)")
ax[1].plot(xn,res[best][1],"--",lw=2,color="#c0392b",
           label=f"Best fit: {best} ($R^2$={res[best][0]:.3f})")
# also overlay GELU fit for reference
if best!="GELU":
    ax[1].plot(xn,res["GELU"][1],":",lw=1.6,color="#27ae60",label=f"GELU ($R^2$={res['GELU'][0]:.3f})")
ax[1].set_xlabel("Normalized input"); ax[1].set_ylabel("Normalized output")
ax[1].set_title("Fit to standard activations"); ax[1].legend(frameon=False); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig_activation.pdf",bbox_inches="tight")
plt.savefig("figures/fig_activation.png",dpi=140,bbox_inches="tight")
print("saved fig_activation")

np.save("figures/act_xn.npy",xn)
np.save("figures/act_yn.npy",yn)
json.dump({"best_fit":best,"r2":float(res[best][0]),
           "r2_gelu":float(res["GELU"][0]),"r2_silu":float(res["SiLU"][0]),
           "Delta":Delta,"Pc_mW":float(Pc*1e3),"Q":Q},
          open("./a1_summary.json","w"))
