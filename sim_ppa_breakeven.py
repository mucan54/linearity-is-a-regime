"""System-level PPA / TOPS-W sensitivity model for a 256x256 optical tensor core.
Honest framing: report a break-even ENVELOPE, not a single optimistic number.
Key structural result: laser/converter/modulator energies amortize over the core's
N^2 MACs and vanish per-MAC; the STATIC thermal-tuning floor amortizes only over
bandwidth (time), so it alone sets whether optics beats electronics.
Pure analytical figure (no GPU). Writes fig_ppa_breakeven.* and fig_ppa_tornado.*"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="."; os.makedirs(os.path.join(OUT,"figures"),exist_ok=True)

# ---------------- Assumption table (nominal + tornado range) ----------------
P = {
 # name          : (nominal, low, high, unit, note)
 "Nch"           : (256,   256,   256,  "",       "tensor-core dimension (256x256)"),
 "B"             : (10e9,  1e9,   100e9,"vec/s",  "operating bandwidth (symbol/vector rate)"),
 "Pheat"         : (1e-3,  1e-4,  20e-3,"W/elem", "static heater power PER tunable element"),
 "k_el"          : (1.0,   0.5,   2.0,  "",       "tunable elements / Nch^2 (crossbar~1, mesh~2)"),
 "eta_wp"        : (0.2,   0.1,   0.3,  "",       "laser wall-plug efficiency"),
 "Pch"           : (13.8e-3,13.8e-3,138e-3,"W/ch", "optical power per input channel. NOT independent of Pact: the\n"
                                                    "                                    self-written premise (Sec. VII) requires the mesh to DELIVER P_c\n"
                                                    "                                    at the bank, so nominal = P_c and the range is mesh loss 0-10 dB"),
 # --- activation ring: INDEPENDENT device parameters. P_c is DERIVED from them.
 # Earlier versions varied P_c and Q as separate tornado bars, which is unphysical:
 # P_c ~ V_eff/Q^2 by construction, so those combinations cannot coexist. The old
 # 0.33 mW lower bound was also a stale artefact (TFLN n2 with a silicon mode volume).
 "n2"            : (1.8e-19,0.9e-19,3.6e-19,"m^2/W","Kerr coefficient of the platform"),
 "Aeff"          : (1.0e-12,0.7e-12,2.0e-12,"m^2",  "effective modal area"),
 "Rring"         : (20e-6, 20e-6, 80e-6,"m",       "ring radius (bend loss pushes R up at high Q)"),
 "Qring"         : (5e5,   1e5,   5e6,  "",        "loaded Q; with n2/Aeff/R this fixes P_c and B_ring"),
 "FoM"           : (10e-15,5e-15, 50e-15,"J/conv-step","ADC/DAC Walden figure of merit"),
 "ENOB"          : (8,     6,     8,    "bits",   "converter effective bits"),
 "Emod"          : (0.5e-12,0.1e-12,1e-12,"J",    "modulator energy per symbol"),
 "Epd"           : (0.1e-12,0.05e-12,0.2e-12,"J", "detector energy per symbol"),
 "E_elec_MAC"    : (0.1e-12,0.03e-12,0.3e-12,"J/MAC","electronic baseline energy per MAC"),
}
nom = {k:v[0] for k,v in P.items()}

NU = 2.99792458e8/1550e-9          # 193.4 THz optical carrier

N0 = 2.21   # TFLN group index

def P_c_of(n2, Aeff, Rring, Qring):
    """Characteristic self-action drive DERIVED from independent device parameters:
    P_c = n0^2 V_eff (2 pi nu) / (4 c n2 Q^2),  V_eff = Aeff * 2 pi R."""
    Veff = Aeff*2*np.pi*Rring
    return (N0**2*Veff*2*np.pi*NU)/(4*2.99792458e8*n2*Qring**2)

def E_opt_per_MAC(B, Pheat, k_el, eta_wp, Pch, FoM, ENOB, Emod, Epd, Nch,
                  n2=1.8e-19, Aeff=1.0e-12, Rring=20e-6, Qring=5e5, **_):
    """Energy per MAC for the optical core.

    IMPORTANT (v23 correction): an activation ring of loaded Q responds at its own
    linewidth B_ring = nu/Q, NOT at the system symbol rate B. Charging the bank as
    Pact/(eta*Nch*B) implicitly assumed one activation per 1/B, i.e. 26x more
    throughput than a Q=5e5 ring can deliver, and disagreed with the per-activation
    figure P_c/B_ring quoted in Sec. VII by exactly B/B_ring.

    Running the bank at B requires M = B/B_ring time-multiplexed replicas. Then
        E_act = (Nch*M*Pact/eta) / (Nch^2 * B) = Pact/(eta*B_ring*Nch)
    and B CANCELS: raising bandwidth buys replicas in proportion. The activation
    bank's energy per MAC is therefore a BANDWIDTH-INDEPENDENT floor.
    The heater floor for the bank scales the same way and goes as Q.
    """
    Pact        = P_c_of(n2, Aeff, Rring, Qring)
    B_ring      = NU/Qring
    replicas    = np.maximum(B/B_ring, 1.0)           # time-multiplexed ring banks
    E_conv_pair = 2*(FoM*(2**ENOB))                   # one ADC + one DAC per dot-product output
    E_thermal   = k_el*Pheat / B                      # mesh heaters: amortize over TIME only
    E_laser     = Pch/(eta_wp*Nch*B)                  # amortizes over Nch^2 MACs
    E_act       = Pact/(eta_wp*B_ring*Nch)            # B-independent activation floor
    E_actheat   = replicas*Pheat/(Nch*B)              # heaters on the replicated bank
    E_conv      = E_conv_pair/Nch
    E_moddet    = (Emod+Epd)/Nch
    return dict(total=E_thermal+E_laser+E_act+E_actheat+E_conv+E_moddet,
                thermal=E_thermal, laser=E_laser, act=E_act, actheat=E_actheat,
                conv=E_conv, moddet=E_moddet)

def tops_w(E_J_per_MAC): return 2.0/(E_J_per_MAC*1e12)   # 2 ops/MAC

# ---- nominal breakdown (sanity) ----
b = E_opt_per_MAC(**nom); Et=b["total"]
print("NOMINAL (B=10GHz, Pheat=1mW): E_opt=%.3f pJ/MAC  -> %.1f TOPS/W  | elec %.1f TOPS/W"%(
      Et*1e12, tops_w(Et), tops_w(nom["E_elec_MAC"])))
print("  breakdown pJ/MAC: mesh-thermal=%.3f laser=%.4f ACT-DRIVE=%.4f act-heat=%.4f conv=%.4f mod+det=%.4f"%(
      b["thermal"]*1e12,b["laser"]*1e12,b["act"]*1e12,b["actheat"]*1e12,b["conv"]*1e12,b["moddet"]*1e12))
print("  activation floor alone = %.3f pJ/MAC = %.1fx the digital baseline (and independent of B)"%(
      b["act"]*1e12, b["act"]/nom["E_elec_MAC"]))

# ================= 1) Break-even envelope: (bandwidth) x (per-element thermal) =================
Bs   = np.logspace(np.log10(0.1e9), np.log10(100e9), 260)      # 0.1 .. 100 GHz
Phs  = np.logspace(np.log10(0.02e-3), np.log10(20e-3), 260)    # 0.02 .. 20 mW
BB, PP = np.meshgrid(Bs, Phs)
kw = dict(nom); 
Egrid = E_opt_per_MAC(B=BB, Pheat=PP, k_el=kw["k_el"], eta_wp=kw["eta_wp"], Pch=kw["Pch"],
                      FoM=kw["FoM"], ENOB=kw["ENOB"], Emod=kw["Emod"], Epd=kw["Epd"], Nch=kw["Nch"],
                      n2=kw["n2"], Aeff=kw["Aeff"], Rring=kw["Rring"], Qring=kw["Qring"])["total"]
TW = tops_w(Egrid)                     # optical TOPS/W over the grid
elec = tops_w(nom["E_elec_MAC"])       # electronic baseline line (nominal)

fig,ax=plt.subplots(figsize=(3.5,3.0))
lv = np.logspace(np.log10(np.nanmin(TW)), np.log10(np.nanmax(TW)), 24)
cf=ax.contourf(Bs/1e9, Phs*1e3, TW, levels=lv, norm=matplotlib.colors.LogNorm(), cmap="viridis")
cb=fig.colorbar(cf,ax=ax); cb.set_label("Optical system TOPS/W", fontsize=7); cb.ax.tick_params(labelsize=6.2)
from matplotlib.ticker import LogLocator, FuncFormatter
cb.locator=LogLocator(base=10.0, subs=(1.0,3.0), numticks=8); cb.formatter=FuncFormatter(lambda v,p: ("%g"%v))
cb.update_ticks()

# break-even contour: optical TOPS/W == electronic baseline.
# With the activation floor included this level is OUTSIDE the plotted range --
# there is no parity anywhere -- so we draw it only if it exists.
has_parity = (TW.max() >= elec)
if has_parity:
    be=ax.contour(Bs/1e9, Phs*1e3, TW, levels=[elec], colors="w", linewidths=1.8)
print("  parity contour exists on this grid: %s (max optical TOPS/W = %.2f vs baseline %.1f)"%(
      has_parity, TW.max(), elec))

# realistic heater bands
ax.axhspan(1,20, color="#c0392b", alpha=0.16)
_bb=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78)
ax.text(0.13, 6.5, "commercial tuners (1\u201320 mW)", color="#a01f12", fontsize=6.3, weight="bold", va="center", bbox=_bb)
ax.axhspan(0.02,0.1, color="#2ca02c", alpha=0.18)
ax.text(0.13, 0.032, "aggressive athermal (<0.1 mW)", color="#12631f", fontsize=6.2, weight="bold", va="center", bbox=_bb)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Operating bandwidth (GHz)", fontsize=8); ax.set_ylabel("Static heater power / element (mW)", fontsize=8)
ax.set_title("The optical core does not beat electronics anywhere", fontsize=8)
ax.text(3.0, 0.045, "activation-drive floor\n%.2f pJ/MAC (B-independent)\nvs %.2f pJ/MAC baseline"%(
        b["act"]*1e12, nom["E_elec_MAC"]*1e12),
        color="#7a1010", fontsize=6.0, ha="center", va="center", weight="bold", bbox=_bb)
ax.tick_params(labelsize=7)
from matplotlib.lines import Line2D
if has_parity:
    _h=[Line2D([],[],color="w",lw=1.8,label="parity vs electronics")]
    ax.legend(handles=_h, loc="lower right", fontsize=5.6, framealpha=0.55, facecolor="#111133",
              edgecolor="none", labelcolor="w", handlelength=1.8, borderpad=0.35)
fig.tight_layout()
fig.savefig(os.path.join(OUT,"figures","fig_ppa_breakeven.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(OUT,"figures","fig_ppa_breakeven.png"),dpi=140,bbox_inches="tight")
print("saved fig_ppa_breakeven")

# ================= 2) Tornado: sensitivity of optical TOPS/W at the nominal point =================
base_TW = tops_w(E_opt_per_MAC(**nom)["total"])
rows=[]
for k,(v0,lo,hi,unit,note) in P.items():
    if k in ("Nch","E_elec_MAC"): 
        if k=="E_elec_MAC": continue   # baseline, not an optical-path param
        if k=="Nch": continue
    kwlo=dict(nom); kwlo[k]=lo; kwhi=dict(nom); kwhi[k]=hi
    tlo=tops_w(E_opt_per_MAC(**kwlo)["total"]); thi=tops_w(E_opt_per_MAC(**kwhi)["total"])
    rows.append((k, tlo, thi, abs(thi-tlo)))
TW_norm = tops_w(E_opt_per_MAC(**nom)["total"]+0.043e-12)
rows.append(("normchg", TW_norm, base_TW, abs(base_TW-TW_norm)))
rows.sort(key=lambda r:r[3])
fig2,ax2=plt.subplots(figsize=(3.5,3.7))
labels={"n2":"Kerr $n_2$ 0.9\u20133.6e-19 m$^2$/W","Aeff":"Mode area 0.7\u20132.0 $\\mu$m$^2$",
        "Rring":"Ring radius 20\u201380 $\\mu$m","Qring":"Loaded $Q$ 1e5\u20135e6 ($\\to P_c$)","normchg":"Norm charge 0\u20130.04 pJ/MAC","B":"Bandwidth 1\u2013100 GHz","Pheat":"Heater/elem 0.1\u201320 mW","k_el":"Elements/Nch\u00b2 0.5\u20132",
        "eta_wp":"Wall-plug 0.1\u20130.3","Pch":"Mesh delivery 14\u2013140 mW/ch (0\u201310 dB)","FoM":"ADC FoM 5\u201350 fJ",
        "ENOB":"ENOB 6\u20138","Emod":"Modulator 0.1\u20131 pJ","Epd":"Detector 0.05\u20130.2 pJ"}
for i,(k,tlo,thi,sw) in enumerate(rows):
    lo,hi=sorted([tlo,thi])
    ax2.barh(i, hi-lo, left=lo, color="#4c72b0", alpha=.85)
    ax2.plot([base_TW,base_TW],[i-0.4,i+0.4]) if False else None
ax2.axvline(base_TW, color="#c0392b", lw=1.8, ls="--", label="nominal optical (%.1f TOPS/W)"%base_TW)
ax2.axvline(TW_norm, color="#e67e22", lw=1.6, ls="-.", label="nominal + norm charge (%.1f)"%TW_norm)
ax2.axvline(elec, color="k", lw=1.5, ls=":", label="electronic baseline (%.0f)"%elec)
ax2.set_yticks(range(len(rows))); ax2.set_yticklabels([labels[k] for k,_,_,_ in rows], fontsize=6.8)
ax2.set_xlabel("Optical TOPS/W at nominal point\n(B=10 GHz, heater=1 mW)", fontsize=7.5)
ax2.set_title("Sensitivity of optical TOPS/W", fontsize=8.5)
ax2.set_xscale("log"); ax2.legend(fontsize=6.2, loc="lower right"); ax2.grid(axis="x",alpha=.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT,"figures","fig_ppa_tornado.pdf"),bbox_inches="tight")
fig2.savefig(os.path.join(OUT,"figures","fig_ppa_tornado.png"),dpi=140,bbox_inches="tight")
print("saved fig_ppa_tornado")

# summary numbers for the paper text
def breakeven_Pheat(B):   # heater power at which E_opt == E_elec, given B
    kw=dict(nom); base=E_opt_per_MAC(B=B,Pheat=0,k_el=kw["k_el"],eta_wp=kw["eta_wp"],Pch=kw["Pch"],FoM=kw["FoM"],ENOB=kw["ENOB"],Emod=kw["Emod"],Epd=kw["Epd"],Nch=kw["Nch"])["total"]
    # E_elec = base + k_el*Pheat/B  ->  Pheat = (E_elec-base)*B/k_el
    return (nom["E_elec_MAC"]-base)*B/kw["k_el"]
for Bg in [1e9,10e9,100e9]:
    print("break-even heater budget at B=%3.0f GHz: %.1f uW/element"%(Bg/1e9, breakeven_Pheat(Bg)*1e6))
for Bg in [1e9,10e9,100e9]:
    kw=dict(nom); b0=E_opt_per_MAC(B=Bg,Pheat=0,k_el=kw["k_el"],eta_wp=kw["eta_wp"],Pch=kw["Pch"],FoM=kw["FoM"],ENOB=kw["ENOB"],Emod=kw["Emod"],Epd=kw["Epd"],Nch=kw["Nch"])["total"]
    print("incl. norm-charge break-even at B=%3.0f GHz: %.1f uW/element"%(Bg/1e9,(nom["E_elec_MAC"]-b0-0.043e-12)*Bg/kw["k_el"]*1e6))
json.dump({"nominal_opt_TOPSW":base_TW,"elec_TOPSW":elec,
           "breakeven_uW_at_1_10_100GHz":[breakeven_Pheat(b)*1e6 for b in (1e9,10e9,100e9)]},
          open(os.path.join(OUT,"ppa_summary.json"),"w"),indent=2)
print("DONE")


# ================= 3) Is there a Q that rescues the activation bank? =================
# P_c ~ Q^-2 and B_ring ~ Q^-1, so the per-activation drive energy P_c/B_ring ~ Q^-1:
# higher Q is cheaper per activation. But keeping up with B needs M = B/B_ring ~ Q
# replicas, so the bank's heater floor ~ Q. The sum has a minimum.
print("\n=== activation-bank optimisation over ring Q ===")
Pc0, Q0 = P_c_of(nom["n2"],nom["Aeff"],nom["Rring"],nom["Qring"]), nom["Qring"]
def bank_terms(Q, Pheat=nom["Pheat"], eta=nom["eta_wp"], N=nom["Nch"]):
    Pc    = Pc0*(Q0/Q)**2                 # P_c ~ Q^-2
    Bring = NU/Q
    return Pc/(eta*Bring*N), (nom["B"]/Bring)*Pheat/(N*nom["B"])
Qg = np.logspace(5.0, 7.5, 6000)
Ea, Eh = bank_terms(Qg); Etot = Ea+Eh
i = int(np.argmin(Etot)); Qstar = Qg[i]
print("  Q* = %.2e  ->  drive %.3f + heater %.3f = %.3f pJ/MAC  (%.1fx baseline)"%(
      Qstar, Ea[i]*1e12, Eh[i]*1e12, Etot[i]*1e12, Etot[i]/nom["E_elec_MAC"]))
print("  at Q*: P_c = %.0f uW, B_ring = %.1f MHz, replicas = %.0f, rings = %.2e"%(
      Pc0*(Q0/Qstar)**2*1e6, NU/Qstar/1e6, nom["B"]/(NU/Qstar), nom["Nch"]*nom["B"]/(NU/Qstar)))
for pitch in (50e-6, 60e-6):
    print("  bank area at %.0f um pitch: %.0f mm^2"%(
          pitch*1e6, nom["Nch"]*nom["B"]/(NU/Qstar)*pitch**2*1e6))
print("  --> even the optimum stays above the %.2f pJ/MAC digital baseline."%(nom["E_elec_MAC"]*1e12))

fig3,ax3=plt.subplots(figsize=(3.45,2.1))
ax3.loglog(Qg, Ea*1e12, lw=1.5, color="#1b3a6b", label=r"drive $P_c/(\eta B_{ring}N)\propto Q^{-1}$")
ax3.loglog(Qg, Eh*1e12, lw=1.5, color="#c0392b", label=r"replica heaters $\propto Q$")
ax3.loglog(Qg, Etot*1e12, lw=2.0, color="#111111", label="total")
ax3.axhline(nom["E_elec_MAC"]*1e12, color="#2ca02c", ls="--", lw=1.3, label="digital baseline")
ax3.plot([Qstar],[Etot[i]*1e12], "o", ms=4, color="#111111")
ax3.annotate("$Q^*\\approx%.1f\\times10^6$, %.2f pJ/MAC"%(Qstar/1e6, Etot[i]*1e12),
             xy=(Qstar, Etot[i]*1e12), xytext=(Qstar*0.028, Etot[i]*1e12*1.9), fontsize=6.0,
             ha="left", va="bottom",
             arrowprops=dict(arrowstyle="-", lw=0.7, color="#444444",
                             shrinkA=1, shrinkB=3))
ax3.set_ylim(Etot[i]*1e12*0.28, None)
ax3.set_xlabel("Activation-ring loaded $Q$", fontsize=7.5)
ax3.set_ylabel("Bank energy (pJ/MAC)", fontsize=7.5)
ax3.set_title("The activation bank has a $Q$ optimum, and it still loses", fontsize=7.8)
ax3.text(0.015, 0.03, "drive curve holds $V_{eff}$ fixed; bend loss\nmakes $R$ grow with $Q$ $\\Rightarrow$ real curve higher",
         transform=ax3.transAxes, fontsize=5.0, ha="left", va="bottom", style="italic", color="#333333")
ax3.tick_params(labelsize=6.4); ax3.grid(alpha=0.3, which="both", lw=0.4)
ax3.legend(fontsize=5.2, frameon=False, loc="upper center", ncol=2,
           columnspacing=1.0, handlelength=1.6, borderpad=0.2)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT,"figures","fig_act_qopt.pdf"),bbox_inches="tight")
fig3.savefig(os.path.join(OUT,"figures","fig_act_qopt.png"),dpi=140,bbox_inches="tight")
print("saved fig_act_qopt")
