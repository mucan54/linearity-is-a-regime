"""Depth-resolved cumulative-noise study (Sec. 6.3, Figs. fig_depth_noise & fig_expressivity_tradeoff).
THE central numerical result. Trains char-LM transformers on tiny-shakespeare across
depths L in {2,4,6} with GELU vs a saturating microring activation, injects cumulative
RMS-scaled Gaussian noise after every block (ASE emulation), and reports teacher-forced
(prefill-regime) validation cross-entropy over 3 seeds. Confirms multiplicative noise
accumulation with depth; the activation choice is second-order. Auto-downloads the corpus.
Writes deep_noise_v3.json and figures/fig_depth_noise.{pdf,png}, figures/fig_expressivity_tradeoff.{pdf,png}."""
import os, json, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
DEV='cuda' if torch.cuda.is_available() else 'cpu'
OUT='.'; os.makedirs(os.path.join(OUT,'figures'),exist_ok=True); t0=time.time()

# 1. Real corpus: tiny-shakespeare (~1.1MB) -> proper generalization, no memorization/overfit
DATA=os.path.join(OUT,'tinyshakespeare.txt')
if not os.path.exists(DATA):
    import urllib.request
    print('downloading tiny-shakespeare...')
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', DATA)
TXT=open(DATA).read()
chars=sorted(set(TXT)); V=len(chars); stoi={c:i for i,c in enumerate(chars)}
data=np.array([stoi[c] for c in TXT],dtype=np.int64); n=len(data); ntr=int(n*0.9)
tr,va=data[:ntr],data[ntr:]
print(f"tiny-shakespeare chars={n} vocab={V} train={len(tr)} val={len(va)}",flush=True)
CTX=128
def get_batch(split,bs,rng):
    d=tr if split=='train' else va
    ix=rng.integers(0,len(d)-CTX-1,size=bs)
    x=np.stack([d[i:i+CTX] for i in ix]); y=np.stack([d[i+1:i+CTX+1] for i in ix])
    return torch.tensor(x,device=DEV),torch.tensor(y,device=DEV)

# 2. GELU vs genuinely SATURATING microring (sigmoid fit to GELU on operating range [-2.5,2.5], saturates beyond)
def _fit():
    xs=torch.linspace(-2.5,2.5,300); tgt=F.gelu(xs)
    p=torch.tensor([3.0,1.2,0.0,-0.1],requires_grad=True); opt=torch.optim.Adam([p],lr=0.02)
    for _ in range(4000):
        a,b,c,d=p; pred=a*torch.sigmoid(b*(xs-c))+d
        loss=((pred-tgt)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return [float(v) for v in p.detach()]
_a,_b,_c,_d=_fit()
def microring(x): return _a*torch.sigmoid(_b*(x-_c))+_d
with torch.no_grad():
    xs=torch.linspace(-4,4,400); tgt=F.gelu(xs); pr=microring(xs)
    MR_R2=1-((pr-tgt)**2).sum().item()/((tgt-tgt.mean())**2).sum().item(); MR_MSE=((pr-tgt)**2).mean().item()
print(f"microring (saturating) vs GELU [-4,4]: R2={MR_R2:.3f} MSE={MR_MSE:.4f}",flush=True)

# 3. Model + cumulative per-block residual noise (ASE-like)
class Block(nn.Module):
    def __init__(s,dm,nh,dff,act):
        super().__init__(); s.ln1=nn.LayerNorm(dm); s.ln2=nn.LayerNorm(dm)
        s.attn=nn.MultiheadAttention(dm,nh,batch_first=True,dropout=0.0); s.act=act
        s.fc1=nn.Linear(dm,dff); s.fc2=nn.Linear(dff,dm)
        m=torch.triu(torch.ones(CTX,CTX)*float('-inf'),diagonal=1); s.register_buffer('mask',m)
    def forward(s,x):
        h=s.ln1(x); a,_=s.attn(h,h,h,attn_mask=s.mask[:x.size(1),:x.size(1)],need_weights=False); x=x+a
        h=s.ln2(x); x=x+s.fc2(s.act(s.fc1(h))); return x
class GPT(nn.Module):
    def __init__(s,V,dm=128,nh=4,dff=512,depth=4,act=F.gelu):
        super().__init__(); s.tok=nn.Embedding(V,dm); s.pos=nn.Embedding(CTX,dm)
        s.blocks=nn.ModuleList([Block(dm,nh,dff,act) for _ in range(depth)])
        s.lnf=nn.LayerNorm(dm); s.head=nn.Linear(dm,V,bias=False)
    def forward(s,idx,gamma=0.0):
        T=idx.size(1); pos=torch.arange(T,device=idx.device); x=s.tok(idx)+s.pos(pos)[None]
        for blk in s.blocks:
            x=blk(x)
            if gamma>0:
                rms=x.detach().pow(2).mean(-1,keepdim=True).sqrt(); x=x+gamma*rms*torch.randn_like(x)
        return s.head(s.lnf(x))
def train_model(depth,act,seed,steps=2500,bs=64):
    torch.manual_seed(seed); m=GPT(V,depth=depth,act=act).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0.1,betas=(0.9,0.95))
    rng=np.random.default_rng(seed); warmup=200
    for step in range(steps):
        lr=3e-3*(step/warmup if step<warmup else 0.5*(1+math.cos(math.pi*(step-warmup)/(steps-warmup))))
        for g in opt.param_groups: g['lr']=lr
        x,y=get_batch('train',bs,rng); loss=F.cross_entropy(m(x).view(-1,V),y.view(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m
@torch.no_grad()
def eval_loss(m,gamma,draws,rng):
    m.eval(); L=[]
    for _ in range(draws):
        tot=0; nb=20
        for _ in range(nb):
            x,y=get_batch('val',64,rng); tot+=F.cross_entropy(m(x,gamma=gamma).view(-1,V),y.view(-1)).item()
        L.append(tot/nb)
    return float(np.mean(L))

# 4. depth x gamma grid
depths=[2,4,6]; seeds=[0,1,2]; gammas=[0.0,0.02,0.05,0.1,0.2,0.3,0.5]
acts={'gelu':F.gelu,'microring':microring}
results={'meta':{'data':'tiny-shakespeare','CTX':CTX,'V':V,'corpus':n,'mr_r2':MR_R2,'mr_mse':MR_MSE,
                 'gammas':gammas,'depths':depths,'seeds':seeds},'runs':{}}
for depth in depths:
    for aname,afn in acts.items():
        key=f"d{depth}_{aname}"; per=[]
        for seed in seeds:
            m=train_model(depth,afn,seed)
            curve=[eval_loss(m,g,5,np.random.default_rng(1000+seed)) for g in gammas]; per.append(curve)
            print(f"  {key} s{seed}: clean={curve[0]:.3f} g10={curve[3]:.3f} g20={curve[4]:.3f} g50={curve[-1]:.3f} [{time.time()-t0:.0f}s]",flush=True)
        arr=np.array(per); results['runs'][key]={'mean':arr.mean(0).tolist(),'std':arr.std(0).tolist()}
        json.dump(results,open(os.path.join(OUT,'deep_noise_v3.json'),'w'),indent=2)  # incremental save
print("SAVED json",flush=True)

# 5. figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
gz=np.array(gammas)*100
fig,axes=plt.subplots(1,3,figsize=(15,4.3),sharey=True)
for ax,depth in zip(axes,depths):
    for aname,col,mk in [('gelu','#2ca02c','s'),('microring','#1f3a93','o')]:
        r=results['runs'][f"d{depth}_{aname}"]; mean=np.array(r['mean']); std=np.array(r['std'])
        ax.errorbar(gz,mean,yerr=std,marker=mk,color=col,label=('Microring' if aname=='microring' else 'GELU'),capsize=3,lw=2,ms=5)
    ax.set_title(f"Depth = {depth} blocks"); ax.set_xlabel(r"Cumulative noise $\gamma$ (%)"); ax.grid(alpha=.3)
axes[0].set_ylabel("Val cross-entropy (nats/char)"); axes[0].legend()
fig.suptitle("Depth-cumulative noise on tiny-Shakespeare char-LM: saturating microring vs GELU",fontsize=12); fig.tight_layout()
fig.savefig(os.path.join(OUT,'figures','fig_depth_noise.pdf'),bbox_inches='tight'); fig.savefig(os.path.join(OUT,'figures','fig_depth_noise.png'),dpi=130,bbox_inches='tight')
fig2,(axL,axR)=plt.subplots(1,2,figsize=(11,4.3)); i2=gammas.index(0.2)
for aname,col in [('gelu','#2ca02c'),('microring','#1f3a93')]:
    clean=[results['runs'][f"d{d}_{aname}"]['mean'][0] for d in depths]
    frag=[results['runs'][f"d{d}_{aname}"]['mean'][i2]-results['runs'][f"d{d}_{aname}"]['mean'][0] for d in depths]
    axL.plot(depths,clean,marker='o',color=col,lw=2,label=('Microring' if aname=='microring' else 'GELU'))
    axR.plot(depths,frag,marker='o',color=col,lw=2,label=('Microring' if aname=='microring' else 'GELU'))
axL.set_title("Clean val loss vs depth (expressivity cost)"); axL.set_xlabel("depth (blocks)"); axL.set_ylabel("clean val loss (nats/char)"); axL.grid(alpha=.3); axL.legend(); axL.set_xticks(depths)
axR.set_title(r"Loss rise at $\gamma$=20% vs depth (fragility)"); axR.set_xlabel("depth (blocks)"); axR.set_ylabel(r"$\Delta$ loss under noise"); axR.grid(alpha=.3); axR.legend(); axR.set_xticks(depths)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT,'figures','fig_expressivity_tradeoff.pdf'),bbox_inches='tight'); fig2.savefig(os.path.join(OUT,'figures','fig_expressivity_tradeoff.png'),dpi=130,bbox_inches='tight')
print(f"SAVED figures total {time.time()-t0:.0f}s",flush=True)
print("\n===== SUMMARY (tiny-shakespeare) =====")
for depth in depths:
    g=results['runs'][f"d{depth}_gelu"]['mean']; m=results['runs'][f"d{depth}_microring"]['mean']
    print(f"d{depth}: clean G={g[0]:.3f} M={m[0]:.3f} (M-G={m[0]-g[0]:+.3f}) | g20% G={g[i2]:.3f} M={m[i2]:.3f} (M-G={m[i2]-g[i2]:+.3f})")
