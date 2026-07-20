"""Component-level energy budget (Sec. 6.2, Fig. fig_energy).
Compares the electronic O/E->activation->E/O round trip against the
all-optical chi(2) path vs bandwidth; locates the ~0.19 GHz break-even.
Writes figures/fig_energy.{pdf,png}."""
import os; os.makedirs("figures",exist_ok=True)

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, json

# ================= Electronic O/E -> activation -> E/O round trip =================
# Representative per-scalar-activation energies (pJ). Ranges from literature-level estimates.
# ADC/DAC via Walden FoM: E = FoM_W * 2^ENOB.  Good FoM_W ~ 10 fJ/conv-step, ENOB=8.
def adc_energy(fom_fJ, enob):   # pJ
    return fom_fJ*1e-3 * (2**enob)
E_pd   = 0.2      # photodetector + TIA
E_adc  = adc_energy(20, 8)   # ~5.1 pJ (mid ADC)
E_dig  = 0.1      # digital LUT nonlinearity
E_dac  = adc_energy(20, 8)   # DAC similar
E_mod  = 0.5      # modulator + driver (E/O)
E_elec = E_pd+E_adc+E_dig+E_dac+E_mod
E_elec_lo = 0.1+adc_energy(5,6)+0.05+adc_energy(5,6)+0.1
E_elec_hi = 0.5+adc_energy(50,8)+0.5+adc_energy(50,8)+1.0
print(f"Electronic round trip: {E_elec:.2f} pJ  (range {E_elec_lo:.2f}-{E_elec_hi:.1f} pJ)")
print(f"  breakdown pJ: PD={E_pd}, ADC={E_adc:.2f}, DIG={E_dig}, DAC={E_dac:.2f}, MOD={E_mod}")

# ================= All-optical chi(2) frequency-conversion activation =================
# Benchmark: Lu et al. Optica 2019 doubly-resonant PPLN-on-TFLN microring:
#   normalized SHG efficiency eta_norm_bench = 2500 /W (=250,000 %/W) at Q_bench ~ 1e6
f0 = 193e12
Q_bench = 1e6
eta_norm_bench = 2500.0   # per Watt
eta_target = 0.30         # target pump->signal conversion driving the nonlinearity
WPE = 0.2                 # pump laser wall-plug efficiency

def analyze(p_scale):
    # eta_norm(Q) ~ eta_norm_bench * (Q/Q_bench)^p   (p=2 conservative singly-res; p=3 doubly-res)
    Q = np.logspace(4, 6, 240)
    B = f0/Q                                   # cavity-linewidth-limited bandwidth (Hz)
    eta_norm = eta_norm_bench*(Q/Q_bench)**p_scale
    P_pump = eta_target/eta_norm               # W needed for target conversion (small-signal)
    E_op_optical = P_pump / B                  # J per linewidth-limited symbol
    E_op_wall = E_op_optical / WPE
    return Q, B, P_pump, E_op_wall

fig,ax=plt.subplots(1,2,figsize=(11.5,4.4))
colors={2:"#1b3a6b",3:"#8e44ad"}
cross={}
for p in (2,3):
    Q,B,Ppump,Ewall = analyze(p)
    ax[0].loglog(B/1e9, Ewall*1e12, lw=2.3, color=colors[p], label=f"all-optical $\\chi^{{(2)}}$ (Q-scaling $p$={p})")
    # crossover with electronic mid
    below = np.where(Ewall*1e12 < E_elec)[0]
    if len(below): cross[p]=B[below[-1]]/1e9   # highest bandwidth still beating electronic
ax[0].axhspan(E_elec_lo, E_elec_hi, color="#c0392b", alpha=0.13)
ax[0].axhline(E_elec, color="#c0392b", lw=1.8, ls="--", label=f"electronic round trip (~{E_elec:.1f} pJ)")
ax[0].set_xlabel("Activation bandwidth (GHz)"); ax[0].set_ylabel("Energy per activation (pJ)")
ax[0].set_title("Energy/op: all-optical vs electronic"); ax[0].grid(alpha=0.3, which="both")
ax[0].legend(frameon=False, fontsize=8.5)

# pump power vs bandwidth
for p in (2,3):
    Q,B,Ppump,Ewall = analyze(p)
    ax[1].loglog(B/1e9, Ppump*1e3, lw=2.3, color=colors[p], label=f"$p$={p}")
ax[1].set_xlabel("Activation bandwidth (GHz)"); ax[1].set_ylabel("Required CW pump power (mW)")
ax[1].set_title(f"Pump power for {int(eta_target*100)}% conversion"); ax[1].grid(alpha=0.3, which="both")
ax[1].legend(frameon=False)
plt.tight_layout()
plt.savefig("figures/fig_energy.pdf", bbox_inches="tight")
plt.savefig("figures/fig_energy.png", dpi=140, bbox_inches="tight")
print("saved fig_energy")

for p in (2,3):
    print(f"p={p}: all-optical beats electronic only up to ~{cross.get(p,float('nan')):.2f} GHz bandwidth")

json.dump({"E_elec_pJ":float(E_elec),"E_elec_lo":float(E_elec_lo),"E_elec_hi":float(E_elec_hi),
           "crossover_GHz_p2":float(cross.get(2,float('nan'))),
           "crossover_GHz_p3":float(cross.get(3,float('nan'))),
           "eta_target":eta_target,"WPE":WPE},
          open("./a2_summary.json","w"))
