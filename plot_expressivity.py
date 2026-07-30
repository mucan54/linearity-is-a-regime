import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=json.load(open("deep_noise_v3.json")); g=R["meta"]["gammas"]; i20=g.index(0.2); depths=[2,4,6]
plt.rcParams.update({"font.size":6.6,"axes.labelsize":6.8,"axes.titlesize":7.0,
                     "xtick.labelsize":6.0,"ytick.labelsize":6.0,"legend.fontsize":5.6})
fig,ax_=plt.subplots(figsize=(2.5,1.85)); ax=[ax_, None]
C={"gelu":"#2ca02c","microring":"#1f3a93"}; M={"gelu":"s","microring":"o"}
for a in ("gelu","microring"):
    clean=[R["runs"][f"d{d}_{a}"]["mean"][0] for d in depths]
    ax[0].plot(depths,clean,marker=M[a],color=C[a],lw=1.5,ms=3.2,mew=0,
               label="GELU" if a=="gelu" else "Microring")
ax[0].set_xlabel("Depth (blocks)"); ax[0].set_ylabel("Clean val loss (nats)")
ax[0].set_title("Expressivity cost of saturation",pad=3); ax[0].grid(alpha=.3,lw=.5)
ax[0].legend(frameon=False); ax[0].set_xticks(depths)
fig.tight_layout(pad=0.3)
fig.savefig("figures/fig_expressivity_tradeoff.pdf",bbox_inches="tight")
fig.savefig("figures/fig_expressivity_tradeoff.png",dpi=150,bbox_inches="tight")
print("expr ok")
