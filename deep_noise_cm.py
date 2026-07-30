# deep_noise_cm.py — common-mode vs independent per-block noise (Sec VI-C scope check)
# Same protocol as deep_noise_dual.py: train once per (depth,seed), evaluate under
# two arms at equal per-feature noise power:
#   'ind': x += g*rms*randn(B,T,C)              (independent, the paper's arm)
#   'cm' : x += g*rms*randn(B,T,1)              (one scalar per token, all features shift together
#                                                = die-level correlated resonance drift)
import os,json,math,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=True
dev='cuda' if torch.cuda.is_available() else 'cpu'
txt=open('/workspace/input.txt','r',encoding='utf-8').read()
chars=sorted(set(txt)); V=len(chars); stoi={c:i for i,c in enumerate(chars)}
data=torch.tensor([stoi[c] for c in txt],dtype=torch.long)
ntr=int(0.9*len(data)); tr,va=data[:ntr],data[ntr:]
def get_batch(split,B,rng):
    d=tr if split=='train' else va
    ix=torch.tensor(rng.integers(0,len(d)-129,size=B))
    x=torch.stack([d[i:i+128] for i in ix]).to(dev)
    y=torch.stack([d[i+1:i+129] for i in ix]).to(dev)
    return x,y
class Blk(nn.Module):
    def __init__(s,C,H):
        super().__init__()
        s.ln1=nn.LayerNorm(C); s.at=nn.MultiheadAttention(C,H,batch_first=True)
        s.ln2=nn.LayerNorm(C); s.mlp=nn.Sequential(nn.Linear(C,4*C),nn.GELU(),nn.Linear(4*C,C))
    def forward(s,x,m):
        h=s.ln1(x); a,_=s.at(h,h,h,attn_mask=m,need_weights=False)
        x=x+a; x=x+s.mlp(s.ln2(x)); return x
class GPT(nn.Module):
    def __init__(s,L,C=128,H=4):
        super().__init__()
        s.emb=nn.Embedding(V,C); s.pos=nn.Embedding(128,C)
        s.blocks=nn.ModuleList([Blk(C,H) for _ in range(L)])
        s.lnf=nn.LayerNorm(C); s.head=nn.Linear(V if False else C,V)
        s.register_buffer('m',torch.triu(torch.ones(128,128)*float('-inf'),diagonal=1))
    def forward(s,idx,gamma=0.0,nm='ind'):
        B,T=idx.shape
        x=s.emb(idx)+s.pos(torch.arange(T,device=idx.device))
        for b in s.blocks:
            x=b(x,s.m[:T,:T])
            if gamma>0:
                rms=x.detach().pow(2).mean(-1,keepdim=True).sqrt()
                if nm=='ind': x=x+gamma*rms*torch.randn_like(x)
                else:         x=x+gamma*rms*torch.randn(x.shape[0],x.shape[1],1,device=x.device)
        return s.head(s.lnf(x))
def eval_loss(m,gamma,nm,draws,rng):
    m.eval(); tot=0
    with torch.no_grad():
        for _ in range(draws):
            x,y=get_batch('val',64,rng)
            tot+=F.cross_entropy(m(x,gamma=gamma,nm=nm).view(-1,V),y.view(-1)).item()
    return tot/draws
depths=[2,4,6]; seeds=[0,1,2]; gammas=[0.0,0.02,0.05,0.1,0.2,0.3,0.5]
res={'meta':{'gammas':gammas,'depths':depths,'seeds':seeds,
     'note':'ind = per-feature independent (paper arm); cm = per-token common-mode scalar (correlated die-level drift), equal per-feature noise power'},'runs':{}}
for L in depths:
    for arm in ('ind','cm'):
        curves=[]
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            m=GPT(L).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
            rng=np.random.default_rng(seed)
            m.train()
            for it in range(1500):
                x,y=get_batch('train',64,rng)
                loss=F.cross_entropy(m(x).view(-1,V),y.view(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            curves.append([eval_loss(m,g,arm,5,np.random.default_rng(1000+seed)) for g in gammas])
        a=np.array(curves)
        res['runs'][f'd{L}_{arm}']={'mean':a.mean(0).tolist(),'std':a.std(0).tolist()}
        json.dump(res,open('/workspace/linearity-is-a-regime/deep_noise_cm.json','w'),indent=1)
        print(f'd{L} {arm} done',flush=True)
print('ALL DONE')
