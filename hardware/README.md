## Step 1: measure block durations (QAE pad + QEC)

Get the matched pad Δ (set by the QAE channel) and the two QEC mid-block durations. Circuit duration on this hardware is set by gate structure (depth, gate count), not by the rotation angles inside it, so this only needs a file describing the ansatz shape (L, opts, form) — create one once:

```bash
python -c "import numpy as np; np.savez('ansatz_shape.npz', L=2, opts=str(dict(anc=3, reset='code', phase=1)), form='selfinv')"
```

### 1a. QAE duration + matched pad:

```bash
python hw_00_dryrun.py --stage estimate \
  --params ansatz_shape.npz \
  --backend ibm_pittsburgh --layout 76,80,82,81 \
  --idle-us 0
```

Read `MATCHED PAD = ... dt = X.XXX us` -> this is Δ. Compute Δ/2.

### 1b. QEC duration:

```bash
python hw_01_deploy.py --backend ibm_pittsburgh --layout 76,80,82,81 \
    --variants qec --qec-mode serial \
    --tau-sweep 0 --shots 1024 --tag probe_ser
```

Write down Δ/2 and the QEC duration. If no duration is printed, use the existing values: Δ/2 = 10.938 us, qec-dur = 10.092 us. the QEC duration is used in **Step 5** `--qec-dur-us`

## Step 2: noise calibration (at the padded $t_{\rm eff}$, not bare τ/2)

Only bare,uncorr run here — no QAE/QEC circuit executes, so the automatic pad never triggers. Each interval idles twice, so passing τ/2 + Δ/2 as `--tau-sweep` makes each interval idle exactly τ/2+Δ/2 — physically identical to "τ/2, pad Δ, τ/2" and matching $t_{\rm eff}$=τ/2+Δ/2 as used in the paper, with no code changes needed.

Desired physical τ/2: 0, 2, 4, 6, 8, 12, 16, 24 us.
Shifted by Δ/2 = 10.938 us: 10.938, 12.938, 14.938, 16.938, 18.938, 22.938, 26.938, 34.938 us.

```bash
python hw_01_deploy.py --backend ibm_pittsburgh --layout 76,80,82,81 \
    --tag cal --outdir cal \
    --variants bare,uncorr \
    --tau-sweep 10.938,12.938,14.938,16.938,18.938,22.938,26.938,34.938 \
    --shots 4096 \
    --spam-ref 1 --drift-check 1 --shuffle 1 --yes
```

Output: `cal/hw_deploy_cal.csv` (read by the trainer), `cal/hw_deploy_cal.npz`.

Check in the CSV:
- $\mu_{hw}$, $\sigma_{deph}$, $\gamma_{hw}$ grow smoothly with `t_eff_us`
- no row has `lam_perp` > 1 (garbage; the trainer drops it) -> rerun if found
- `t_eff_us` equals the shifted `--tau-sweep` values exactly (confirms the pad was correctly folded into the sweep, not that the run is unpadded)

## Step 3: train the QAE offline

```bash
python train_qae_multipoint.py \
  --tau-list 0,2,4,6,8,12,16,24 \
  --measured-csv cal/hw_deploy_cal.csv \
  --rho-corr 0.05 --L 2 \
  --out dephasing_multipoint_L2.npz
```

`--tau-list` must be the physical τ/2 values (0,2,4,...), not the shifted calibration sweep. The trainer fits smooth curves against the CSV's t_eff_us column and evaluates them at `--tau-list`, so this is correct regardless of the shift.

Output: `dephasing_multipoint_L2.npz`, `dephasing_multipoint_L2.csv` (summary).

## Step 4: lock the parameters for hardware

```bash
python hw_00_dryrun.py --stage scan-multi \
  --npz-multi dephasing_multipoint_L2.npz --L 2 \
  --out-multi hw_params_multipoint_L2.npz
```

Must print: `[t1] [t2] [t3] -> OK`, `[B] best RMS < 1e-5`, every `P0..P7 line -> OK`. If any line says FAIL, stop.

Output: `hw_params_multipoint_L2.npz` (needed by Step 5).

## Step 5: the real benchmark

4 qubits: 81,80,76,82 (ancilla=82). 

Use the QEC duration measured in **Step 1b.** for `--qec-dur-us`.

```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  python hw_01_deploy.py --backend ibm_pittsburgh --tag r1_$i \
    --outdir runs/r1_multipoint \
    --variants qae_clean,uncorr,bare,qec \
    --params-clean-multi hw_params_multipoint_L2.npz \
    --qec-mode serial --qec-dur-us 10.092 \
    --tau-sweep 0,2,4,6,8,12,16,24 --layout 76,80,82,81 --shots 1024 \
    --spam-ref 1 --drift-check 1 --shuffle 1 --seed $((1234+i)) --yes
done
```

## Step 6: plot results

```bash
python hw_02_aggregate.py --dir runs/r1_multipoint --tag r1_avg --drift-tol 0.03
```