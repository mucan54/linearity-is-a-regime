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
 "Pch"           : (0.5e-3,0.1e-3,2e-3, "W/ch",   "optical power per input channel"),
 "FoM"           : (10e-15,5e-15, 50e-15,"J/conv-step","ADC/DAC Walden figure of merit"),
 "ENOB"          : (8,     6,     8,    "bits",   "converter effective bits"),
 "Emod"          : (0.5e-12,0.1e-12,1e-12,"J",    "modulator energy per symbol"),
 "Epd"           : (0.1e-12,0.05e-12,0.2e-12,"J", "detector energy per symbol"),
 "E_elec_MAC"    : (0.1e-12,0.03e-12,0.3e-12,"J/MAC","electronic baseline energy per MAC"),
}
nom = {k:v[0] for k,v in P.items()}

def E_opt_per_MAC(B, Pheat, k_el, eta_wp, Pch, FoM, ENOB, Emod, Epd, Nch, **_):
    E_conv_pair = 2*(FoM*(2**ENOB))                 # one ADC + one DAC per dot-product output
    E_thermal = k_el*Pheat / B                       # <-- amortizes over TIME only (NOT over MACs)
    E_laser   = Pch/(eta_wp*Nch*B)                   # amortizes over Nch^2 MACs
    E_conv    = E_conv_pair/Nch                       # one converter per Nch MACs
    E_moddet  = (Emod+Epd)/Nch                        # one mod+det per Nch MACs
    return dict(total=E_thermal+E_laser+E_conv+E_moddet,
                thermal=E_thermal, laser=E_laser, conv=E_conv, moddet=E_moddet)

def tops_w(E_J_per_MAC): return 2.0/(E_J_per_MAC*1e12)   # 2 ops/MAC

# ---- nominal breakdown (sanity) ----
b = E_opt_per_MAC(**nom); Et=b["total"]
print("NOMINAL (B=10GHz, Pheat=1mW): E_opt=%.3f pJ/MAC  -> %.1f TOPS/W  | elec %.1f TOPS/W"%(
      Et*1e12, tops_w(Et), tops_w(nom["E_elec_MAC"])))
print("  breakdown pJ/MAC: thermal=%.3f laser=%.4f conv=%.4f mod+det=%.4f"%(
      b["thermal"]*1e12,b["laser"]*1e12,b["conv"]*1e12,b["moddet"]*1e12))

# ================= 1) Break-even envelope: (bandwidth) x (per-element thermal) =================
Bs   = np.logspace(np.log10(0.1e9), np.log10(100e9), 260)      # 0.1 .. 100 GHz
Phs  = np.logspace(np.log10(0.02e-3), np.log10(20e-3), 260)    # 0.02 .. 20 mW
BB, PP = np.meshgrid(Bs, Phs)
kw = dict(nom); 
Egrid = E_opt_per_MAC(B=BB, Pheat=PP, k_el=kw["k_el"], eta_wp=kw["eta_wp"], Pch=kw["Pch"],
                      FoM=kw["FoM"], ENOB=kw["ENOB"], Emod=kw["Emod"], Epd=kw["Epd"], Nch=kw["Nch"])["total"]
TW = tops_w(Egrid)                     # optical TOPS/W over the grid
elec = tops_w(nom["E_elec_MAC"])       # electronic baseline line (nominal)

fig,ax=plt.subplots(figsize=(7.8,5.2))
lv = np.logspace(np.log10(np.nanmin(TW)), np.log10(np.nanmax(TW)), 24)
cf=ax.contourf(Bs/1e9, Phs*1e3, TW, levels=lv, norm=matplotlib.colors.LogNorm(), cmap="viridis")
cb=fig.colorbar(cf,ax=ax,label="Optical system TOPS/W"); 
# break-even contour: optical TOPS/W == electronic baseline
be=ax.contour(Bs/1e9, Phs*1e3, TW, levels=[elec], colors="w", linewidths=2.4)
ax.clabel(be, fmt={elec:"break-even vs electronics (%.0f TOPS/W)"%elec}, fontsize=8)
# realistic heater bands
ax.axhspan(1,20, color="#c0392b", alpha=0.16)
ax.text(0.13, 4.2, "commercial thermal tuners (1\u201320 mW/elem)", color="#c0392b", fontsize=8.4, weight="bold")
ax.axhspan(0.02,0.1, color="#2ca02c", alpha=0.18)
ax.text(0.13, 0.045, "aggressive athermal (<0.1 mW)", color="#1e7a1e", fontsize=8.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Operating bandwidth  (GHz)"); ax.set_ylabel("Static heater power per element  (mW)")
ax.set_title("Where does the optical tensor core beat electronics?\n(256$\\times$256; representative parameters)", fontsize=10.5)
ax.annotate("optics wins\n(only here)", xy=(60,0.06), color="w", fontsize=9, ha="center", weight="bold")
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
rows.sort(key=lambda r:r[3])
fig2,ax2=plt.subplots(figsize=(7.4,4.3))
labels={"B":"Bandwidth 1\u2013100 GHz","Pheat":"Heater/elem 0.1\u201320 mW","k_el":"Elements/Nch\u00b2 0.5\u20132",
        "eta_wp":"Wall-plug 0.1\u20130.3","Pch":"Opt power/ch 0.1\u20132 mW","FoM":"ADC FoM 5\u201350 fJ",
        "ENOB":"ENOB 6\u20138","Emod":"Modulator 0.1\u20131 pJ","Epd":"Detector 0.05\u20130.2 pJ"}
for i,(k,tlo,thi,sw) in enumerate(rows):
    lo,hi=sorted([tlo,thi])
    ax2.barh(i, hi-lo, left=lo, color="#4c72b0", alpha=.85)
    ax2.plot([base_TW,base_TW],[i-0.4,i+0.4]) if False else None
ax2.axvline(base_TW, color="#c0392b", lw=1.8, ls="--", label="nominal optical (%.1f TOPS/W)"%base_TW)
ax2.axvline(elec, color="k", lw=1.5, ls=":", label="electronic baseline (%.0f)"%elec)
ax2.set_yticks(range(len(rows))); ax2.set_yticklabels([labels[k] for k,_,_,_ in rows], fontsize=8.6)
ax2.set_xlabel("Optical system TOPS/W at nominal point (B=10 GHz, heater=1 mW)")
ax2.set_title("Sensitivity of optical TOPS/W (tornado)", fontsize=10.5)
ax2.set_xscale("log"); ax2.legend(fontsize=8, loc="lower right"); ax2.grid(axis="x",alpha=.3)
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
json.dump({"nominal_opt_TOPSW":base_TW,"elec_TOPSW":elec,
           "breakeven_uW_at_1_10_100GHz":[breakeven_Pheat(b)*1e6 for b in (1e9,10e9,100e9)]},
          open(os.path.join(OUT,"ppa_summary.json"),"w"),indent=2)
print("DONE")
