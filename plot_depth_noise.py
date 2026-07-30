import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=json.load(open("deep_noise_all.json")); g=np.array(R["meta"]["gammas"])*100
plt.rcParams.update({"font.size":6.8,"axes.labelsize":7.2,"axes.titlesize":7.6,
                     "xtick.labelsize":6.4,"ytick.labelsize":6.4,"legend.fontsize":5.9})
fig,ax=plt.subplots(figsize=(3.45,2.25))
cols={2:"#bcd4f2",4:"#8ab4e8",6:"#4a86c8",8:"#2f6fb5",12:"#0d2547"}
for d in (2,4,6,8,12):
    for a,ls,mk in [("gelu","-","o"),("microring","--","s")]:
        r=R["runs"][f"d{d}_{a}"]; m=np.array(r["mean"]); s=np.array(r["std"])
        ax.plot(g,m,ls,marker=mk,color=cols[d],lw=1.5,ms=2.8,mew=0)
        ax.fill_between(g,m-s,m+s,color=cols[d],alpha=0.13,lw=0)
ax.axvspan(0.13, 0.68, color="#d9a441", alpha=0.35, lw=0, zorder=0)
_y0,_y1=ax.get_ylim()
ax.text(5.5, _y0+0.03*(_y1-_y0), "ASE range, TFLN\n(0.13\u20130.68%:\n14 mW ring,\n0.5 mW/ch core)", fontsize=5.4, ha="left", va="bottom",
        color="#6b4e0f", linespacing=1.05)
ax.set_xlabel(r"Cumulative per-block noise $\gamma$ (%)")
ax.set_ylabel("Val cross-entropy (nats/char)")
ax.set_title("Depth sets fragility; activation does not",pad=4)
ax.grid(alpha=0.3,lw=0.5)
h1=[Line2D([],[],color=cols[d],lw=1.6,label=f"{d}") for d in (2,4,6,8,12)]
h2=[Line2D([],[],color="0.35",lw=1.4,ls="-",marker="o",ms=3,mew=0,label="GELU"),
    Line2D([],[],color="0.35",lw=1.4,ls="--",marker="s",ms=3,mew=0,label="Microring")]
lg=ax.legend(handles=h1,loc="upper left",frameon=False,ncol=5,columnspacing=0.5,handlelength=0.9,title="depth $L$",title_fontsize=5.6)
ax.add_artist(lg); ax.legend(handles=h2,loc="lower right",frameon=False,handlelength=2.0)
fig.tight_layout(pad=0.3)
import os as _os; _os.makedirs("figures",exist_ok=True)
fig.savefig("figures/fig_depth_noise.pdf",bbox_inches="tight"); fig.savefig("figures/fig_depth_noise.png",dpi=150,bbox_inches="tight")
print("depth ok")
