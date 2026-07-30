import os, json, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
DEV='cuda'; OUT='.'; t0=time.time()
TXT=open(os.path.join(OUT,'tinyshakespeare.txt')).read()
chars=sorted(set(TXT)); V=len(chars); stoi={c:i for i,c in enumerate(chars)}
data=np.array([stoi[c] for c in TXT],dtype=np.int64); n=len(data); ntr=int(n*0.9)
tr,va=data[:ntr],data[ntr:]
CTX=128
def get_batch(split,bs,rng):
    d=tr if split=='train' else va
    ix=rng.integers(0,len(d)-CTX-1,size=bs)
    x=np.stack([d[i:i+CTX] for i in ix]); y=np.stack([d[i+1:i+CTX+1] for i in ix])
    return torch.tensor(x,device=DEV),torch.tensor(y,device=DEV)
def _fit():
    xs=torch.linspace(-2.5,2.5,300); tgt=F.gelu(xs)
    p=torch.tensor([3.0,1.2,0.0,-0.1],requires_grad=True); opt=torch.optim.Adam([p],lr=0.02)
    for _ in range(4000):
        a,b,c,d=p; pred=a*torch.sigmoid(b*(xs-c))+d
        loss=((pred-tgt)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return [float(v) for v in p.detach()]
_a,_b,_c,_d=_fit()
def microring(x): return _a*torch.sigmoid(_b*(x-_c))+_d
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
    def forward(s,idx,gamma=0.0,nm='const'):
        T=idx.size(1); pos=torch.arange(T,device=idx.device); x=s.tok(idx)+s.pos(pos)[None]
        for blk in s.blocks:
            x=blk(x)
            if gamma>0:
                rms=x.detach().pow(2).mean(-1,keepdim=True).sqrt()
                if nm=='const':
                    x=x+gamma*rms*torch.randn_like(x)
                else:  # sqrt: per-element std ~ sqrt(|x_i|), equal mean noise power per token
                    ax=x.detach().abs()
                    scale=(ax/ax.mean(-1,keepdim=True).clamp_min(1e-8)).sqrt()
                    x=x+gamma*rms*scale*torch.randn_like(x)
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
def eval_loss(m,gamma,nm,draws,rng):
    m.eval(); L=[]
    for _ in range(draws):
        tot=0; nb=20
        for _ in range(nb):
            x,y=get_batch('val',64,rng); tot+=F.cross_entropy(m(x,gamma=gamma,nm=nm).view(-1,V),y.view(-1)).item()
        L.append(tot/nb)
    return float(np.mean(L))
depths=[2,4,6]; seeds=[0,1,2]; gammas=[0.0,0.02,0.05,0.1,0.2,0.3,0.5]
acts={'gelu':F.gelu,'microring':microring}
results={'meta':{'gammas':gammas,'depths':depths,'seeds':seeds,'note':'same trained model evaluated under const-relative vs sqrt-scaled (equal mean noise power) arms'},'runs':{}}
for depth in depths:
    for aname,afn in acts.items():
        per={'const':[],'sqrt':[]}
        for seed in seeds:
            m=train_model(depth,afn,seed)
            for nm in ['const','sqrt']:
                curve=[eval_loss(m,g,nm,5,np.random.default_rng(1000+seed)) for g in gammas]
                per[nm].append(curve)
            print(f"d{depth}_{aname} s{seed}: const g20={per['const'][-1][4]:.3f} g50={per['const'][-1][-1]:.3f} | sqrt g20={per['sqrt'][-1][4]:.3f} g50={per['sqrt'][-1][-1]:.3f} [{time.time()-t0:.0f}s]",flush=True)
        for nm in ['const','sqrt']:
            arr=np.array(per[nm]); results['runs'][f"d{depth}_{aname}_{nm}"]={'mean':arr.mean(0).tolist(),'std':arr.std(0).tolist()}
        json.dump(results,open(os.path.join(OUT,'deep_noise_dual.json'),'w'),indent=2)
print("DUAL DONE",flush=True)
