# 10-Replica LSC-IVR Diamond Analysis

## Run

- System: 1728-atom, 6x6x6 diamond C
- Potential: C_Tersoff_1989
- Temperature: 300 K
- Replica seeds: 12345--12354
- Trajectory: 10,000,000 steps, 0.5 fs
- HAC grid: 0.5--999.5 ps, 1000 rows
- Hessian: one shared 5184-dimensional eigenvector file
- Process status: all 10 replicas completed with return code 0

## Weighted Result

The merged `hac.out` uses log-sum-exp normalized Wigner weights.

| Window | kx | ky | kz | isotropic average |
|---|---:|---:|---:|---:|
| Full 1 ns | 2005 | 5505 | 2863 | 3458 |
| Last 500 ps | 1667 | 6841 | 2483 | 3664 |
| Last 250 ps | 1581 | 7235 | 2449 | 3755 |
| Endpoint | 1417 | 5655 | 3670 | 3581 |

Units are W/m/K. These are running Green-Kubo values and are not a converged
bulk conductivity estimate.

## Weight Diagnostic

| seed | normalized weight |
|---:|---:|
| 12345 | 1.1875e-6 |
| 12346 | 1.2126e-18 |
| 12347 | 4.1606e-7 |
| 12348 | 5.0119e-23 |
| 12349 | 6.6654e-13 |
| 12350 | 6.1318e-6 |
| 12351 | 4.4820e-5 |
| 12352 | 2.3202e-10 |
| 12353 | 0.9999473 |
| 12354 | 1.3776e-7 |

- Effective replicas: `N_eff = 1.000105`
- Maximum normalized weight: `0.9999473`
- The weighted average is therefore statistically equivalent to nearly one
  trajectory, despite having 10 completed trajectories.

For comparison, the same 10 trajectories combined with equal weights give:

- Full-window average: `3808 W/m/K`
- Last-half average: `4017 W/m/K`
- Endpoint average: `4113 W/m/K`

The difference is caused by the importance weights, not by a HAC merge error.
The merged HAC agrees with an independent weighted reduction to about
`3e-12` in the computed kappa columns.

## Interpretation

The 10 replica run validates that independent seeds produce distinct HAC
curves and that Wigner weighting is applied in the process-level merge. It
does not provide a reliable ten-sample uncertainty estimate because the
anharmonic reweighting has collapsed the effective sample size to one.

The running conductivity still drifts between the full, last-half, last-
quarter, and endpoint estimates. The 2.14 nm cell and one effective weighted
sample are insufficient for a converged diamond thermal conductivity.

For a statistically useful reweighted result, increase the replica count
substantially and monitor `N_eff`; alternatively report the equal-weight
LSC-IVR ensemble separately when the importance-sampling estimator is too
degenerate. Do not interpret the small `hac_uncertainty.csv` values as a
physical error bar when `N_eff` is approximately one.
