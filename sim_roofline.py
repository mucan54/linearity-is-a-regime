"""Roofline visualization of the workload bound (Sec. 5.6, Fig. fig_roofline).
Shows that LLM autoregressive decode has low arithmetic intensity and is pinned
to the SHARED memory-bandwidth roof, so a raised optical compute ceiling gives
no benefit there; only high-reuse prefill/training reaches the compute-bound
regime. The argument depends only on workload arithmetic intensity, NOT on the
(illustrative) ceiling heights. Pure analytical/matplotlib figure -- no GPU.
Writes figures/fig_roofline.{pdf,png}."""
import os; os.makedirs("figures", exist_ok=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Representative parameters (illustrative; the conclusion is ceiling-independent) ----
BW_TBps   = 3.0        # shared memory bandwidth ~ HBM (TB/s). slope P[TOPS] = BW * I
P_ELEC    = 1.0e3      # electronic digital peak (TOPS), representative
P_OPT     = 1.0e4      # optical compute ceiling (TOPS), representative
slope     = lambda I: BW_TBps * I           # TB/s * ops/byte -> TOPS
ridge_e   = P_ELEC / BW_TBps                 # ~333 ops/byte
ridge_o   = P_OPT  / BW_TBps                 # ~3333 ops/byte

I = np.logspace(-1, 4, 500)
roof_e = np.minimum(P_ELEC, slope(I))
roof_o = np.minimum(P_OPT,  slope(I))

fig, ax = plt.subplots(figsize=(7.6, 5.0))
# shared memory-bandwidth roof (sloped)
ax.plot(I, slope(I), color="#888", lw=1.4, ls="--", zorder=1)
ax.text(2.3, slope(2.3)*1.5, "shared memory-bandwidth roof  (P = BW $\\times$ I)",
        color="#666", fontsize=8.3, rotation=30, rotation_mode="anchor")
# rooflines
ax.plot(I, roof_o, color="#1f3a93", lw=2.6, label="Optical compute ceiling (this work, representative)", zorder=3)
ax.plot(I, roof_e, color="#2ca02c", lw=2.6, label="Electronic digital (GPU/TPU, representative)", zorder=3)

# ridge points
for r, P, c in [(ridge_e, P_ELEC, "#2ca02c"), (ridge_o, P_OPT, "#1f3a93")]:
    ax.plot([r], [P], "o", color=c, ms=6, zorder=4)

# workload markers
I_dec, I_pre = 1.5, 5.0e3
# decode: on the shared slope (both substrates coincide)
ax.axvline(I_dec, color="#c0392b", lw=1.0, ls=":", zorder=2)
ax.plot([I_dec], [slope(I_dec)], "s", color="#c0392b", ms=8, zorder=5)
ax.annotate("LLM decode (batch 1)\nlow intensity $\\Rightarrow$ memory-bound:\nboth substrates pinned to the same\nroof; optical ceiling gives no gain",
            xy=(I_dec, slope(I_dec)), xytext=(0.115, 20),
            fontsize=8.4, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
# prefill/training: compute-bound, optical ceiling helps
ax.axvline(I_pre, color="#7d3c98", lw=1.0, ls=":", zorder=2)
ax.plot([I_pre], [P_ELEC], "s", color="#2ca02c", ms=8, zorder=5)
ax.plot([I_pre], [P_OPT],  "s", color="#1f3a93", ms=8, zorder=5)
ax.annotate("", xy=(I_pre, P_OPT), xytext=(I_pre, P_ELEC),
            arrowprops=dict(arrowstyle="<->", color="#444", lw=1.3))
ax.text(I_pre*1.15, np.sqrt(P_ELEC*P_OPT), "optical\nheadroom", fontsize=8.5, color="#444", va="center")
ax.annotate("prefill / training\n(high operand reuse)\n$\\Rightarrow$ compute-bound",
            xy=(I_pre, P_OPT), xytext=(3.2e2, 2.4e4),
            fontsize=8.6, color="#7d3c98",
            arrowprops=dict(arrowstyle="->", color="#7d3c98", lw=1.2))

# regime labels
ax.text(0.5, 4.5e4, "memory-bound", fontsize=9, color="#999", style="italic")
ax.text(3.3e3, 1.35, "compute-bound", fontsize=9, color="#999", style="italic")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.1, 1e4); ax.set_ylim(1, 1e5)
ax.set_xlabel("Arithmetic intensity  (operations / byte)")
ax.set_ylabel("Attainable performance  (TOPS)")
ax.set_title("Roofline: why optical compute helps prefill but not autoregressive decode", fontsize=10.5)
ax.grid(True, which="both", alpha=0.18)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=8.3, framealpha=0.95)
fig.tight_layout()
fig.savefig("figures/fig_roofline.pdf", bbox_inches="tight")
fig.savefig("figures/fig_roofline.png", dpi=140, bbox_inches="tight")
print(f"saved fig_roofline  | ridge_elec={ridge_e:.0f} ridge_opt={ridge_o:.0f} ops/byte  | decode I={I_dec} prefill I={I_pre:.0f}")
