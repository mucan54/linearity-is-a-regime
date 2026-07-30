# Linearity Is a Regime, Not a Law

Reproducibility code for the paper **"Linearity Is a Regime, Not a Law: Frequency
Degrees of Freedom as Native Nonlinearity Primitives for Optical Computing"**
(M. C. Özdemir, independent researcher).

The paper argues that the maxim *"optics is linear"* describes the weak-field
χ⁽¹⁾ **regime**, not a law: the χ⁽²⁾E² and χ⁽³⁾E³ terms that generate new
frequencies are exactly the terms that make the effective transfer function
nonlinear, so frequency degrees of freedom can serve as native nonlinearity
primitives. This repo holds the numerical proof-of-concept simulations behind
the paper's figures. Everything here is **idealized modeling — there is no
fabricated device.**

## Platform and the P_c correction (v22)

All power figures are quoted for **thin-film lithium niobate**. The platform is
forced, not chosen: mechanism (i) requires χ⁽²⁾, which excludes centrosymmetric
silicon and silicon nitride, and TFLN is also two-photon-absorption-free at
1550 nm.

Revisions before v22 quoted `P_c ≈ 16 µW` (silicon) and then `≈ 0.33 mW` (TFLN
n₂, but **silicon mode volume**). Both are wrong for this device. `V_eff` must be
built from the actual geometry: with `A_eff ≈ 1 µm²` and `R ≈ 20 µm`,
`V_eff ≈ 1.3e-16 m³` and **`P_c ≈ 14 mW`**.

`sim_microring_activation.py` now asserts the self-consistency condition that
catches this class of error — the coupled-mode normalisation requires the Kerr
index shift to reach half a linewidth at `P_in ~ P_c`, i.e. `Δn = n₀/2Q`, and the
buildup calculation must reproduce it:

    SELF-CONSISTENCY  dn_actual/dn_required = 1.10

The old `V_eff` returned 0.03. The assert fails if a future edit reintroduces an
inconsistent geometry.

Downstream anchors, all re-derived by the scripts: ENOB ≈ 11.7 bits at 10 GHz;
ASE γ ≈ 0.13 % at `P_c` against 0.68 % at 0.5 mW/channel; self-written parameter
cost `P_c/B ≈ 36 pJ` at the ring's 386 MHz bandwidth (≈ 0.9 nJ at converter rate)
against ≈ 3 pJ to write the same parameter externally; XPM ≈ 5 mrad on a 1 mm bus
but ≈ 16 rad inside the ring. `sim_ppa_breakeven.py` now carries the activation
bank's optical drive as an explicit term (`Pact`), ≈ 0.027 pJ/MAC at 10 GHz,
which is what removes the sub-GHz break-even corner entirely.

## The activation-drive floor (v23)

`sim_ppa_breakeven.py` previously charged the activation bank as
`Pact/(eta*Nch*B)`, which implicitly assumes one activation per `1/B`. A ring of
loaded Q responds at its **own** linewidth `B_ring = nu/Q ≈ 386 MHz`, not at the
system symbol rate — so that form disagreed with the per-activation figure
`P_c/B_ring ≈ 36 pJ` quoted in §VII by exactly `B/B_ring ≈ 26`.

Running the bank at B needs `M = B/B_ring` time-multiplexed replicas, so

    E_act = (Nch*M*P_c/eta) / (Nch^2 * B) = P_c/(eta * B_ring * Nch)

and **B cancels**: bandwidth buys replicas in proportion. The activation bank is
a *bandwidth-independent* floor of **≈0.70 pJ/MAC**, seven times the 0.1 pJ/MAC
digital baseline. There is no parity anywhere on the (B, P_heat) plane — the
script now checks this and refuses to draw a contour that does not exist.

The one escape is ring Q: `P_c ∝ Q⁻²` and `B_ring ∝ Q⁻¹` make the drive term
`∝ Q⁻¹`, but the replica count and hence the bank's heater floor go `∝ Q`. The
sum minimises at `Q* ≈ 4×10⁶` at ≈0.17 pJ/MAC — still above baseline, and needing
≈5.5×10⁴ rings (≈35–50 mm²) for one 256-wide layer. `figures/fig_act_qopt.*`.

**Also fixed:** `plot_depth_noise.py` and `plot_expressivity.py` were writing
their PDFs to the repo root instead of `figures/`, so `figures/fig_depth_noise.*`
had been stale since the ASE band was first revised. Both now write to `figures/`.

## Common-mode drift arm (v27)

`deep_noise_cm.py` adds a third noise arm to the §VI-C protocol: a per-token
**common-mode scalar** (same shift on every feature, equal per-feature noise
power) modelling correlated die-level resonance drift, against the paper's
independent arm. Result (`deep_noise_cm.json`): the common-mode arm is
indistinguishable from the clean run to six decimal places at every depth and
every γ up to 50%. The immunity is structural — every consumption point in a
pre-norm transformer passes a LayerNorm, whose mean-subtraction kills a
feature-uniform offset exactly (machine precision) and whose variance division
kills a common gain to O(ε). What this arm does **not** capture: a real
resonance shift deforms the activation curve, and the first-order residue
ε·(∂T/∂Δ)(x)·x is x-dependent and survives normalization — flagged in the
paper as unmodeled.

## Held-out extrapolation test (v28)

`analysis_heldout.py` measures the scaling law's real out-of-range error: fit
on L≤8 only, predict the held-out L=12. Under-prediction is −18..−24% for one
~1.5× range step; compounded to L=96 that is ×3–4, so the paper's central
extrapolation band is quoted as a floor (lifted band ≲0.003 nats — verdict
unchanged, ASE remains far from binding).

## Contents

| Script | Paper | Output |
|---|---|---|
| `sim_microring_activation.py` | §6.1 | `figures/fig_activation.*` — Kerr microring drop-port self-action curve, fit to sigmoid (R²=0.986) / GELU (0.889) / SiLU (0.885), operated just below the √3 bistability threshold. Also writes `figures/act_{xn,yn}.npy`. |
| `sim_energy_budget.py` | §6.2 | `figures/fig_energy.*` — electronic O/E→act→E/O (~11 pJ) vs all-optical χ⁽²⁾ path vs bandwidth; the all-optical path wins only below ≈0.19 GHz. |
| `sim_ppa_breakeven.py` | §6.4 | `figures/fig_ppa_breakeven.*`, `figures/fig_ppa_tornado.*` — system-level TOPS/W break-even for a 256×256 optical tensor core; shows the static thermal-tuning floor (which alone fails to amortize over MACs) decides whether optics beats electronics, plus a sensitivity tornado. |
| `sim_roofline.py` | §5.6 | `figures/fig_roofline.*` — roofline showing autoregressive decode is memory-bound (pinned to the shared memory roof, so the optical compute ceiling gives no gain) while only high-reuse prefill/training reach the compute-bound regime an optical ceiling can lift. |
| `sim_architecture_diagram.py` | §4 | `figures/fig_architecture.*` — schematic of the proposed frequency-native nonlinearity pipeline (no computation). |
| `deep_noise_study.py` | §6.3 | `figures/fig_depth_noise.*`, `figures/fig_expressivity_tradeoff.*` — **the central result** (see below). |
| `plot_depth_noise.py` (paper figure), `plot_expressivity.py` (supplementary; the expressivity panel was dropped from the final paper) | §6.3 (figures) | Re-plot `figures/fig_depth_noise.*` and `figures/fig_expressivity_tradeoff.*` from `deep_noise_v3.json` in the single-column form used in the paper (depth by colour, activation by line style). |
| `sim_softmax_cascade.py` | §4.2 | Compares two ways to synthesize the softmax exponential from ring cascades: externally detuned stages (1.8% worst-case error at N=3, 4.0 dB loss, improving with N) versus self-action stages solved self-consistently from the paper's cubic (14–17%, not improving). Supports the scoping claim that self-action shapes the activation, not the exponential. |
| `deep_noise_deep.py` | §6.3 (depth extension) | `deep_noise_all.json` — extends the depth sweep to L = 8 and 12, giving the five-depth fit ΔL = c·γ^a·L^b with a = 2.07 ± 0.04, b = 0.73 ± 0.04 (the L ≤ 6 fit alone reads b = 0.58 and under-predicts L = 12). |
| `deep_noise_dual.py` | §6.3 (noise-model robustness) | `deep_noise_dual.json` — same protocol; each model trained once, evaluated under BOTH the constant-relative arm and a √|x|-scaled (intensity-encoding ASE) arm at equal mean injected noise power. The √ arm is marginally harsher at depth (≤0.05 nats), leaving every conclusion unchanged. |
| `sim_singleblock_noise_probe.py` | §6.3 (superseded) | `figures/fig_noise.*` — the misleading shallow probe; kept only for transparency. |

## The central result (§6.3)

`deep_noise_study.py` trains character-level transformers on tiny-Shakespeare
(~1.1M chars, held-out validation) across depths **L ∈ {2, 4, 6}**, with GELU
versus a genuinely **saturating microring activation** (R²=0.930 to GELU over its
operating range, saturating beyond). To emulate amplified-spontaneous-emission
(ASE) accumulation, zero-mean Gaussian noise of relative std **γ** — scaled per
token by the residual-stream RMS — is injected **after every block**, so a depth-L
network accumulates L insertions. We report teacher-forced validation
cross-entropy in the **compute-bound / prefill regime** (not autoregressive
decode, which the paper places out of optical reach), averaged over 3 seeds.

Three findings, all honest to a fault:

1. **Noise accumulates multiplicatively with depth** (confirms §5.3): the
   noise-induced loss rise at γ=20% grows **0.09 → 0.15 → 0.22 nats** from
   L=2 → 4 → 6; at γ=50% validation loss climbs ≈2.1 → 2.6 → 3.1 nats.
2. **No activation robustness advantage:** GELU and the saturating microring are
   statistically indistinguishable under noise (differences ≤0.014 nats, within
   the seed-to-seed scatter of ≈0.02–0.03).
3. **Small expressivity cost:** the clean-loss gap is ≤0.007 nats and *shrinks*
   with depth — the network compensates for the weaker pointwise map.

The single-block probe (`sim_singleblock_noise_probe.py`) instead suggests the
microring is *more* noise-robust. That is a shallow-depth, task-ceiling artifact
that does **not** survive the depth-resolved study; it is included solely to make
the reversal transparent.

![depth vs noise](figures/fig_depth_noise.png)
![expressivity/fragility](figures/fig_expressivity_tradeoff.png)

## Requirements

Python ≥ 3.10 and:

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib, torch
```

A CUDA GPU is recommended for the two PyTorch scripts, but they fall back to CPU.
Developed and run on an NVIDIA GB10 (Grace-Blackwell) box; nothing is
hardware-specific.

## Reproduce every figure

```bash
python sim_microring_activation.py     # writes act_*.npy used by the probe
python sim_energy_budget.py
python sim_architecture_diagram.py
python sim_roofline.py
python sim_ppa_breakeven.py
python deep_noise_study.py             # auto-downloads tiny-shakespeare (~1 MB)
python sim_singleblock_noise_probe.py  # optional, superseded
```

Figures land in `figures/`. Reference outputs are already committed there.

## Citation

Paper under preparation; an arXiv link will be added here on posting.

## License

MIT — see [`LICENSE`](LICENSE).
