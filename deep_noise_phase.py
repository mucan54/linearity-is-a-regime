"""deep_noise_phase.py -- does the ring's intensity-dependent PHASE matter? (referee item 4)

The paper models the activation as a real, amplitude-only saturating map |t(U)|.
The physical device returns a COMPLEX coefficient t(U) = 1/(1 + i(u-Delta)) whose
argument sweeps ~100 deg over the operating range. Downstream the field enters a
coherent mesh, so what the next linear stage actually multiplies is t(U), not |t(U)|.

With a real-weighted mesh W and homodyne readout, Re(W z) = W Re(z), so the
physically realised activation is Re t(u) = |t| cos(arg t), NOT |t|. This script
trains under the paper's amplitude-only assumption and then evaluates the SAME
weights under:
    amp          -- |t(u)|          (what the paper assumes)
    cplx         -- Re t(u)         (what a coherent mesh + homodyne actually sees)
    cplx_static  -- Re[e^{-i phibar} t(u)]  (single global static phase compensation)
and additionally trains natively under 'cplx' to separate train-deploy mismatch
from an intrinsic loss of expressivity.
"""
import os, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
DELTA = 1.6                      # operating detuning, just below sqrt(3)

def ring_u(x, iters=40):
    """Stable root of u[(Delta-u)^2+1] = x  (single-valued for Delta < sqrt3)."""
    u = x / (DELTA*DELTA + 1.0)                       # small-signal start
    for _ in range(iters):
        g  = u**3 - 2*DELTA*u**2 + (DELTA*DELTA+1)*u - x
        gp = 3*u**2 - 4*DELTA*u + (DELTA*DELTA+1)
        u  = u - g/gp.clamp(min=1e-6)
        u  = u.clamp(min=0.0)
    return u

def ring_apply(f, mode, phibar=0.0):
    """f: real signed field amplitude. Power x=f^2 drives the ring."""
    x = f*f
    u = ring_u(x)
    d = u - DELTA                       # delta_eff/(gamma_t/2)
    mod = 1.0/torch.sqrt(1.0 + d*d)     # |t|
    if mode == 'amp':
        return mod*f
    arg = -torch.atan(d)                # arg t
    return mod*torch.cos(arg - phibar)*f

txt = open('/workspace/input.txt', 'r', encoding='utf-8').read()
chars = sorted(set(txt)); V = len(chars); stoi = {c:i for i,c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in txt], dtype=torch.long)
ntr = int(0.9*len(data)); tr, va = data[:ntr], data[ntr:]
def get_batch(split, B, rng):
    d = tr if split=='train' else va
    ix = torch.tensor(rng.integers(0, len(d)-129, size=B))
    return (torch.stack([d[i:i+128] for i in ix]).to(dev),
            torch.stack([d[i+1:i+129] for i in ix]).to(dev))

class Blk(nn.Module):
    def __init__(s, C, H):
        super().__init__()
        s.ln1=nn.LayerNorm(C); s.at=nn.MultiheadAttention(C,H,batch_first=True)
        s.ln2=nn.LayerNorm(C); s.fc1=nn.Linear(C,4*C); s.fc2=nn.Linear(4*C,C)
    def forward(s, x, m, mode, phibar):
        h=s.ln1(x); a,_=s.at(h,h,h,attn_mask=m,need_weights=False); x=x+a
        z=s.fc1(s.ln2(x))
        return x + s.fc2(ring_apply(z, mode, phibar))

class GPT(nn.Module):
    def __init__(s, L, C=128, H=4):
        super().__init__()
        s.emb=nn.Embedding(V,C); s.pos=nn.Embedding(128,C)
        s.blocks=nn.ModuleList([Blk(C,H) for _ in range(L)])
        s.lnf=nn.LayerNorm(C); s.head=nn.Linear(C,V)
        s.register_buffer('m', torch.triu(torch.ones(128,128)*float('-inf'), diagonal=1))
    def forward(s, idx, mode='amp', phibar=0.0):
        B,T=idx.shape
        x=s.emb(idx)+s.pos(torch.arange(T,device=idx.device))
        for b in s.blocks: x=b(x, s.m[:T,:T], mode, phibar)
        return s.head(s.lnf(x))

def evaluate(m, mode, phibar, draws, seed):
    m.eval(); rng=np.random.default_rng(seed); tot=0.0
    with torch.no_grad():
        for _ in range(draws):
            x,y=get_batch('val',64,rng)
            tot+=F.cross_entropy(m(x,mode,phibar).view(-1,V), y.view(-1)).item()
    return tot/draws

def train(L, mode, seed, steps=1500, phibar=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    m=GPT(L).to(dev); opt=torch.optim.AdamW(m.parameters(), lr=3e-4)
    rng=np.random.default_rng(seed); m.train()
    for _ in range(steps):
        x,y=get_batch('train',64,rng)
        loss=F.cross_entropy(m(x,mode,phibar).view(-1,V), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m

# choose the static compensation as the mean arg t over a plausible operating range
_x = torch.linspace(0.0, 4.0, 4001)
_u = ring_u(_x); _d = _u - DELTA
PHIBAR = float((-torch.atan(_d)).mean())
SWING  = float((-torch.atan(_d)).max() - (-torch.atan(_d)).min())
print(f'phase swing over x in [0,4]: {math.degrees(SWING):.1f} deg, mean arg = {math.degrees(PHIBAR):.1f} deg', flush=True)

res={'meta':{'Delta':DELTA,'phibar_rad':PHIBAR,'phase_swing_deg':math.degrees(SWING),
             'note':'trained under amp (paper assumption), evaluated under amp/cplx/cplx_static; '
                    'plus a natively cplx-trained control'},'runs':{}}
for L in (2,4):
    for seed in (0,1,2):
        m=train(L,'amp',seed)
        r={'amp'        : evaluate(m,'amp',0.0,5,1000+seed),
           'cplx'       : evaluate(m,'cplx',0.0,5,1000+seed),
           'cplx_static': evaluate(m,'cplx',PHIBAR,5,1000+seed)}
        mn=train(L,'cplx',seed,phibar=PHIBAR)
        r['cplx_native']=evaluate(mn,'cplx',PHIBAR,5,1000+seed)
        res['runs'][f'd{L}_s{seed}']=r
        json.dump(res, open('/workspace/linearity-is-a-regime/deep_noise_phase.json','w'), indent=1)
        print(f'L={L} seed={seed}: {r}', flush=True)
print('ALL DONE', flush=True)
