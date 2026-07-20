"""SUPERSEDED single-block noise probe (discussed in Sec. 6.3 as the misleading
shallow baseline). Kept for transparency: on an easy classification task a
one-layer net shows the microring activation as *more* noise-robust than GELU,
an artifact that does NOT survive the depth-resolved study (deep_noise_study.py).
Requires figures/act_{xn,yn}.npy -> run sim_microring_activation.py first.
Writes figures/fig_noise.{pdf,png}."""
import os; os.makedirs("figures",exist_ok=True)

import numpy as np, torch, torch.nn as nn, json, math
torch.manual_seed(0); np.random.seed(0)
dev="cuda" if torch.cuda.is_available() else "cpu"
V,L,NCLS=10,12,6                      # 6 balanced bins on the digit-sum
def raw(n):
    x=torch.randint(0,V,(n,L)); return x, x.sum(1)
Xtr,Str=raw(60000); Xte,Ste=raw(10000)
qs=np.quantile(Str.numpy(),[i/NCLS for i in range(1,NCLS)])
tolabel=lambda s: torch.bucketize(s, torch.tensor(qs,dtype=s.dtype))
Ytr=tolabel(Str); Yte=tolabel(Ste)
print("NCLS",NCLS,"balance",[int((Ytr==k).sum()) for k in range(NCLS)])
Xtr,Ytr,Xte,Yte=[t.to(dev) for t in (Xtr,Ytr,Xte,Yte)]
xn=np.load("figures/act_xn.npy"); yn=np.load("figures/act_yn.npy")
xn_t=torch.tensor(xn,dtype=torch.float32,device=dev); yn_t=torch.tensor(yn,dtype=torch.float32,device=dev)

class MicroringAct(nn.Module):
    def __init__(self,noise_gamma=0.0):
        super().__init__(); self.noise_gamma=noise_gamma
        z=torch.linspace(-3,3,241); g=0.5*z*(1+torch.tanh(math.sqrt(2/math.pi)*(z+0.044715*z**3)))
        best=None
        for a_in in torch.linspace(0.1,0.6,24):
            for b_in in torch.linspace(0.0,1.0,24):
                u=(a_in*z+b_in).clamp(0,1); ring=torch.tensor(np.interp(u.numpy(),xn,yn),dtype=torch.float32)
                A=torch.stack([ring,torch.ones_like(ring)],1); sol,_,_,_=torch.linalg.lstsq(A,g.unsqueeze(1))
                err=((A@sol).squeeze()-g).pow(2).mean().item()
                if best is None or err<best[0]: best=(err,float(a_in),float(b_in),sol.squeeze().tolist())
        self.err,self.a_in,self.b_in,self.aout=best
    def forward(self,z):
        u=(self.a_in*z+self.b_in).clamp(0,1)
        idx=torch.searchsorted(xn_t,u.clamp(min=float(xn_t[0]),max=float(xn_t[-1]))).clamp(1,len(xn_t)-1)
        x0,x1=xn_t[idx-1],xn_t[idx]; y0,y1=yn_t[idx-1],yn_t[idx]; ring=y0+(y1-y0)*(u-x0)/(x1-x0+1e-9)
        if self.noise_gamma>0 and not self.training: ring=ring+self.noise_gamma*ring.std()*torch.randn_like(ring)
        return self.aout[0]*ring+self.aout[1]
class GELUn(nn.Module):
    def __init__(self,noise_gamma=0.0): super().__init__(); self.noise_gamma=noise_gamma; self.g=nn.GELU()
    def forward(self,z):
        a=self.g(z)
        if self.noise_gamma>0 and not self.training: a=a+self.noise_gamma*a.std()*torch.randn_like(a)
        return a
class TinyTF(nn.Module):
    def __init__(self,act_factory):
        super().__init__(); d=32
        self.emb=nn.Embedding(V,d); self.pos=nn.Parameter(torch.randn(1,L,d)*0.02); self.act=act_factory()
        self.blocks=nn.ModuleList([nn.ModuleDict(dict(ln1=nn.LayerNorm(d),attn=nn.MultiheadAttention(d,2,batch_first=True),
            ln2=nn.LayerNorm(d),fc1=nn.Linear(d,64),fc2=nn.Linear(64,d))) for _ in range(1)])
        self.head=nn.Linear(d,NCLS)
    def forward(self,x):
        h=self.emb(x)+self.pos
        for b in self.blocks:
            hn=b["ln1"](h); a,_=b["attn"](hn,hn,hn); h=h+a
            h=h+b["fc2"](self.act(b["fc1"](b["ln2"](h))))
        return self.head(h[:,-1])           # last-token readout (reduced redundancy)
def train_model(f,epochs=30):
    m=TinyTF(f).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); lf=nn.CrossEntropyLoss(); bs=512
    for ep in range(epochs):
        m.train(); perm=torch.randperm(len(Xtr),device=dev)
        for i in range(0,len(Xtr),bs):
            idx=perm[i:i+bs]; opt.zero_grad(); loss=lf(m(Xtr[idx]),Ytr[idx]); loss.backward(); opt.step()
        sch.step()
    return m
@torch.no_grad()
def acc(m): m.eval(); return (m(Xte).argmax(1)==Yte).float().mean().item()
m_gelu=train_model(lambda:GELUn(0.0)); acc_gelu=acc(m_gelu)
m_opt=train_model(lambda:MicroringAct(0.0)); acc_opt=acc(m_opt)
print(f"GELU {acc_gelu*100:.2f}%  Microring(fitMSE={m_opt.act.err:.4f}) {acc_opt*100:.2f}%")
gammas=[0.0,0.05,0.10,0.20,0.30,0.50,0.75,1.0,1.5,2.0]
def sweep(m):
    o=[]
    for g in gammas: m.act.noise_gamma=g; o.append(float(np.mean([acc(m) for _ in range(8)])))
    m.act.noise_gamma=0.0; return o
ao=sweep(m_opt); ag=sweep(m_gelu)
print("opt ",[round(a*100,2) for a in ao]); print("gelu",[round(a*100,2) for a in ag])
def da(a,base,d):
    for g,v in zip(gammas,a):
        if base-v>=d: return g
    return None
g1=da(ao,acc_opt,0.01); g5=da(ao,acc_opt,0.05)
print(f"Microring 1% drop @gamma~{g1}, 5% @gamma~{g5}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11.5,4.3))
zz=np.linspace(-3,3,200); gl=0.5*zz*(1+np.tanh(np.sqrt(2/np.pi)*(zz+0.044715*zz**3)))
with torch.no_grad(): ml=m_opt.act(torch.tensor(zz,dtype=torch.float32,device=dev)).cpu().numpy()
ax[0].plot(zz,gl,lw=2,color="#27ae60",label="GELU (target)"); ax[0].plot(zz,ml,"--",lw=2.2,color="#1b3a6b",label=f"Microring act (fit MSE={m_opt.act.err:.3f})")
ax[0].set_xlabel("pre-activation $z$"); ax[0].set_ylabel("activation output"); ax[0].set_title("(a) Simulated optical activation vs GELU"); ax[0].legend(frameon=False); ax[0].grid(alpha=0.3)
ax[1].plot([g*100 for g in gammas],[a*100 for a in ao],"o-",lw=2.2,color="#1b3a6b",label="Microring activation")
ax[1].plot([g*100 for g in gammas],[a*100 for a in ag],"s--",lw=2,color="#27ae60",label="GELU (ideal)")
ax[1].axhline(100/NCLS,color="grey",ls=":",lw=1,label=f"chance ({100/NCLS:.1f}%)")
ax[1].axvspan(0,10,color="#2ecc71",alpha=0.08); ax[1].text(2,30,"realistic\noptical SNR",fontsize=8,color="#27632a")
ax[1].set_xlabel("Relative activation noise $\\gamma$ (%)"); ax[1].set_ylabel("Test accuracy (%)"); ax[1].set_title("(b) Accuracy vs analog (ASE-equiv.) noise"); ax[1].legend(frameon=False,fontsize=8.5); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figures/fig_noise.pdf",bbox_inches="tight"); plt.savefig("figures/fig_noise.png",dpi=140,bbox_inches="tight")
print("saved fig_noise")
json.dump({"acc_gelu":acc_gelu,"acc_opt":acc_opt,"fit_mse":m_opt.act.err,"gammas":gammas,"acc_opt_noise":ao,"acc_gelu_noise":ag,"g1_opt":g1,"g5_opt":g5,"NCLS":NCLS},open("./a3_summary.json","w"))
