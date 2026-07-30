"""Held-out extrapolation test for the depth-noise scaling law (Sec. VI-C).
Standard practice before trusting a power law outside its fitted range: fit on
L in {2,4,6,8} only, predict the held-out L=12, and measure the real one-step
extrapolation error. Requires deep_noise_all.json (produced on the GX10).
Result used in the paper: -18..-24% under-prediction per ~1.5x range step;
compounded over the five-fold log-range to L=96 this multiplies the
extrapolated figure by ~3-4x, so the central band is a floor."""
import json, numpy as np
d=json.load(open("deep_noise_all.json"))
gam=d["meta"]["gammas"]
def pts(arm, Ls, gmax=0.101):
    X=[];Y=[]
    for L in Ls:
        m=d["runs"][f"d{L}_{arm}"]["mean"]; clean=m[0]
        for gi,g in enumerate(gam):
            if 0<g<=gmax:
                X.append((g,L)); Y.append(m[gi]-clean)
    return np.array(X), np.array(Y)
def fit(X,Y):
    A=np.column_stack([np.ones(len(X)), np.log(X[:,0]), np.log(X[:,1])])
    coef,*_=np.linalg.lstsq(A, np.log(Y), rcond=None)
    return np.exp(coef[0]),coef[1],coef[2]
for arm in ("microring","gelu"):
    c,a,b=fit(*pts(arm,[2,4,6,8]))
    Xte,Yte=pts(arm,[12]); pred=c*Xte[:,0]**a*Xte[:,1]**b
    err=(pred-Yte)/Yte*100
    print(f"{arm:9s} held-out (fit L<=8 -> predict L=12): a={a:.3f} b={b:.3f}")
    for (g,L),p,y,e in zip(Xte,pred,Yte,err):
        print(f"   g={g:.2f}: pred={p:.4f} actual={y:.4f} err={e:+.1f}%")
import math
step=math.log(96/12)/math.log(12/8)
print(f"\ncompounded over {step:.1f} steps to L=96: x{(1/(1-0.18))**step:.1f} (18%) .. x{(1/(1-0.24))**step:.1f} (24%)")
