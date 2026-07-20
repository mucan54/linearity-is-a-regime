"""Proposed frequency-native nonlinearity pipeline (Sec. 4, Fig. fig_architecture).
Pure-matplotlib schematic; no computation. Writes figures/fig_architecture.{pdf,png}."""
import os; os.makedirs("figures",exist_ok=True)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size":9.5})
fig,ax=plt.subplots(figsize=(12,4.6)); ax.set_xlim(0,12); ax.set_ylim(0,5); ax.axis("off")

def box(x,y,w,h,text,fc,ec="#222",fs=9.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=fc,ec=ec,lw=1.4))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,wrap=True)
def arrow(x1,y1,x2,y2,color="#333",style="-|>",lw=1.8,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=14,
                                 lw=lw,color=color,ls=ls))

# optical-domain band
ax.add_patch(FancyBboxPatch((0.15,1.5),9.4,3.15,boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="#eaf3fb",ec="#8fb8e0",lw=1.2,ls="--"))
ax.text(4.85,4.5,"OPTICAL DOMAIN  (field stays coherent — no O/E)",ha="center",fontsize=9,color="#2a5d94",style="italic")

box(0.4,2.6,1.5,1.0,"Frequency\ncomb /\nmulti-λ laser","#fdf2d0")
box(2.3,2.6,1.7,1.0,"WDM encode\n$x_i \\to \\omega_i$\n(data on planes)","#dbe6f5")
box(4.4,2.35,1.9,1.5,"Linear core\nMZI / ring MVM\n$y=Wx$\n(per-plane)","#d5e8d4")
box(6.7,2.6,2.6,1.0,"Nonlinear ring bank\nKerr self-action  $\\omega_r(\\,|x|^2)$\nactivation per plane","#f8d7da")

# softmax/mixing sub-branch
box(6.7,1.7,2.6,0.7,"$\\chi^{(2)}$ mixing:  $\\omega_i\\!\\pm\\!\\omega_j$\nexp / normalize (softmax)","#f3e0f7",fs=8.6)

# detector + electronics
box(10.0,2.6,1.7,1.0,"Photodetector\narray  $|E|^2$","#e2e2e2")
box(10.0,0.7,1.7,1.0,"Electronic\ncontrol / bias\n(LayerNorm, loop)","#e2e2e2")

arrow(1.9,3.1,2.3,3.1)
arrow(4.0,3.1,4.4,3.1)
arrow(6.3,3.1,6.7,3.1)
arrow(9.3,3.1,10.0,3.1)
# nonlinear -> mixing
arrow(8.0,2.6,8.0,2.4,color="#8e44ad")
# detector -> electronics (O/E boundary, red dashed)
arrow(10.85,2.6,10.85,1.7,color="#c0392b",ls="--")
ax.text(11.15,2.15,"O/E tax",color="#c0392b",fontsize=8,rotation=90,va="center")
# electronics feedback to encode (dashed)
arrow(10.0,1.2,3.2,1.2,color="#888",ls="--",lw=1.3,style="-|>")
ax.text(6.4,1.32,"electronic feedback / reprogramming (bias, gains)",color="#666",fontsize=7.6,ha="center")

# frequency-plane inset annotation
ax.text(3.15,2.35,"$\\{\\omega_1,\\omega_2,\\dots\\}$ = separable computational planes",
        fontsize=7.8,color="#2a5d94",ha="center")

ax.set_title("Frequency-native photonic transformer layer: linear MVM + frequency-encoded nonlinearity",
             fontsize=10.5,pad=8)
plt.tight_layout()
plt.savefig("figures/fig_architecture.pdf",bbox_inches="tight")
plt.savefig("figures/fig_architecture.png",dpi=140,bbox_inches="tight")
print("saved fig_architecture")
