"""Microring self-action activation (Sec. 6.1, Fig. fig_activation).
Solves the normalized Kerr coupled-mode model just below the bistability
threshold and fits the drop-port transfer to sigmoid/GELU/SiLU.
Writes figures/fig_activation.{pdf,png} and figures/act_{xn,yn}.npy
(the latter are consumed by sim_singleblock_noise_probe.py)."""
import os; os.makedirs("figures",exist_ok=True)

import numpy as np
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib as _mpl
_mpl.rcParams.update({"font.size":6.6,"axes.labelsize":6.8,"axes.titlesize":7.0,"xtick.labelsize":6.0,"ytick.labelsize":6.0,"legend.fontsize":5.4,"lines.linewidth":1.4})

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

# ---------- Physical anchoring: characteristic Kerr power P_c ----------
# PLATFORM: thin-film lithium niobate. Forced, not chosen -- mechanism (i) needs
# chi(2), which excludes centrosymmetric Si and SiN, and TFLN is TPA-free at 1550 nm.
# NOTE (correction, v22): earlier revisions carried Veff = 3e-18 m^3, a SILICON
# geometry (Aeff ~ 0.1 um^2 x L ~ 30 um). For a TFLN mode of Aeff ~ 1 um^2 that
# same Veff implies L = 3 um, i.e. R = 0.48 um -- impossible at Q = 5e5. Veff is
# now built from the actual geometry and cross-checked below.
lam0=1550e-9; c=2.99792458e8; nu0=c/lam0; w0=2*np.pi*nu0
Q    = 5e5
n0   = 2.21        # TFLN extraordinary index at 1550 nm
n2   = 1.8e-19     # m^2/W, TFLN Kerr (~25x below silicon's 4.5e-18)
Aeff = 1.0e-12     # m^2, ~1 um^2 ridge mode
Rring= 20e-6       # m, ring radius (smallest that plausibly holds Q = 5e5)
Lring= 2*np.pi*Rring
Veff = Aeff*Lring
Pc   = (n0**2 * Veff * w0)/(4*c*n2*Q**2)
print(f"TFLN ring: Aeff={Aeff*1e12:.2f} um^2, R={Rring*1e6:.0f} um, L={Lring*1e6:.1f} um, Veff={Veff:.2e} m^3")
print(f"Characteristic Kerr power P_c ~ {Pc*1e3:.2f} mW  (Q={Q:.0e})")
print(f"Operating input range: {xin.max()*Pc*1e3:.1f} mW  (x_in_max={xin.max():.1f} * P_c)")

# ---------- Self-consistency check (this is what caught the Veff error) ----------
# The CMT normalisation demands that at P_in ~ P_c the Kerr index shift reach
# half a linewidth: dn_required = n0/(2Q).  Compute dn independently from the
# resonant buildup and compare -- the two must agree to O(1) or Veff is wrong.
linewidth = nu0/Q
FSR       = 1.19e12
finesse   = FSR/linewidth
P_circ    = Pc*finesse/np.pi          # resonant power buildup
dn_actual = n2*(P_circ/Aeff)
dn_req    = n0/(2*Q)
print(f"  linewidth={linewidth/1e6:.0f} MHz, finesse={finesse:.0f}, P_circ={P_circ:.2f} W")
print(f"  SELF-CONSISTENCY  dn_actual/dn_required = {dn_actual/dn_req:.2f}  "
      f"(dn_actual={dn_actual:.2e}, dn_required={dn_req:.2e})")
assert 0.5 < dn_actual/dn_req < 2.0, "Veff/Aeff inconsistent with the CMT normalisation"

# ---------- Thermo-optic vs Kerr (Sec. VI-A) ----------
# Ratio is INDEPENDENT of drive level: circulating power cancels.
dndT = 3.3e-5      # K^-1, LN thermo-optic coefficient
for alpha in (0.02,0.05,0.10):
    for Rth in (1e3,1e4,1e5):
        ratio = dndT*alpha*Rth*Aeff*np.pi/(n2*finesse)
        print(f"  dn_th/dn_Kerr @ alpha={alpha:.0%}, Rth={Rth:.0e} K/W : {ratio:8.1f}x")

# ---------- Downstream anchors quoted in the paper ----------
h=6.62607015e-34; hnu=h*nu0
for B,lbl in ((1e10,"10 GHz"),(linewidth,"ring linewidth")):
    N=Pc/(hnu*B)
    print(f"  ENOB @ {lbl:14s}: {np.log2(np.sqrt(N)):.1f} bits ; write energy P_c/B = {Pc/B*1e12:.1f} pJ")
Q_hi = nu0/1e10                       # Q that passes a 10 GHz signal
print(f"  at B=10 GHz need Q={Q_hi:.2e} -> P_c={Pc*(Q/Q_hi)**2:.2f} W, write={Pc*(Q/Q_hi)**2/1e10*1e12:.0f} pJ")
print(f"  heater 1-20 mW vs P_c: x{1e-3/Pc:.2f} to x{20e-3/Pc:.2f}")
print(f"  ASE gamma at P_c: {0.68*np.sqrt(0.5e-3/Pc):.2f} %  (0.68% at 0.5 mW/ch)")
gnl = 2*np.pi*n2/(lam0*Aeff)
phi_bus = 2*gnl*1e-3*255*Pc
print(f"  gamma_nl={gnl:.2f} /W/m ; XPM bus (255 ch @ P_c, 1 mm)={phi_bus*1e3:.1f} mrad ; in-ring={phi_bus*finesse:.1f} rad")

fig,ax=plt.subplots(1,2,figsize=(3.45,1.95))
ax[0].plot(xin, xout, lw=1.6, color="#1b3a6b")
ax[0].set_xlabel(r"Input $x_{in}$ ($P_c$)")
ax[0].set_ylabel("Drop output")
ax[0].set_title(f"Ring transfer ($\\Delta$={Delta})",pad=3)
ax[0].grid(alpha=0.3)
ax[1].plot(xn,yn,lw=1.7,color="#1b3a6b",label="Microring")
ax[1].plot(xn,res[best][1],"--",lw=2,color="#c0392b",
           label=f"{best} ($R^2$={res[best][0]:.3f})")
# also overlay GELU fit for reference
if best!="GELU":
    ax[1].plot(xn,res["GELU"][1],":",lw=1.6,color="#27ae60",label=f"GELU ($R^2$={res['GELU'][0]:.3f})")
ax[1].set_xlabel("Norm. input"); ax[1].set_ylabel("Norm. output")
ax[1].set_title("Fit to activations",pad=3); ax[1].legend(frameon=False); ax[1].grid(alpha=0.3)
plt.tight_layout(pad=0.25)
plt.savefig("figures/fig_activation.pdf",bbox_inches="tight")
plt.savefig("figures/fig_activation.png",dpi=140,bbox_inches="tight")
print("saved fig_activation")

np.save("figures/act_xn.npy",xn)
np.save("figures/act_yn.npy",yn)
json.dump({"best_fit":best,"r2":float(res[best][0]),
           "r2_gelu":float(res["GELU"][0]),"r2_silu":float(res["SiLU"][0]),
           "Delta":Delta,"Pc_mW":float(Pc*1e3),"Q":Q},
          open("./a1_summary.json","w"))
