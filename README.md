# visvar

Simulate **time-variable radio sources** (pulsars, periodic/modulated emitters,
flares, arbitrary lightcurves) as visibilities in **Measurement Sets**.

**Code was developed with assistance from Claude.**

Two modes:

* `visvar.simulate(sky, obs, "out.ms")` – build a new MS from an array, times,
  channels and a phase centre.
* `visvar.inject(ms, sky, column="DATA", mode="add")` – add sources to an
  existing (real or template) MS, using its own UVW/TIME/INTERVAL/channels.

## Install

```bash
pip install .            # needs python-casacore, astropy, numpy
```

## Quick start

```python
import numpy as np, visvar as vv
from astropy.coordinates import SkyCoord
from astropy.time import Time

array = vv.Array.from_ms("../data/atca.ms")         # build array from an existing measurement set
pc = SkyCoord("10h00m00s", "-60d00m00s")
obs = vv.Observation(
  array, pc, Time("2026-03-01T20:00:00", scale="utc"),
  duration=600, dt=1.0, freqs=np.linspace(7e8, 1e9, 300),
  corrs=("XX", "XY", "YX", "YY")
)

psr = vv.Source(
  SkyCoord("10h01m30s", "-59d57m00s"),
  I=vv.PulseTrain(
    period=76, width=0.08, peak=0.8,
    dm=300.0, ref_freq=1e9
  ),
  spectral_index=-1.5, ref_freq=1e9
)
vv.simulate([psr], obs, "psr.ms", sefd=450.0, seed=1)
```

## Concepts

**Lightcurves** (`visvar.lightcurves`) give flux(t, $\nu$) per Stokes parameter.
Time `t` is seconds since the observation start (keeps phase precision for
ms pulsars). Built-ins: `Constant`, `Sinusoid`, `PulseTrain` (Gaussian / von
Mises / boxcar / custom profile, `pdot`, dispersion measure, baseline),
`Tabulated` (optionally periodic), `Burst` (dispersible),
`FromCallable`. They support `+` and scalar `*`.

**Sources** carry `I, Q, U, V` lightcurves (floats are constant), a power-law
spectral index and a position (`SkyCoord` or `(ra_deg, dec_deg)`). A **Sky** is
a list of sources; static field sources are just constant lightcurves.

**Integration and channel averaging.** The lightcurve is *averaged* over each
dump and channel, not sampled at the midpoint. `PulseTrain` and `Sinusoid`
integrate analytically/exactly (any number of periods per dump); dispersive
channel smearing is handled by sub-dividing channels. Other lightcurves are
sub-sampled automatically from their `timescale`; override with
`nt_sub=` / `nf_sub=`.

**Polarisation:** Ideal feeds: linear (`XX XY YX YY`) or circular
(`RR RL LR LL`), or any subset, from Stokes I/Q/U/V (Have not tested this part yet; use with caution.).

**Noise:** adds Gaussian noise with per-component ${\rm rms} = {\rm SEFD} / \sqrt{2\Delta\nu\tau}$ per correlation; `WEIGHT`/`SIGMA` are filled to match.

**UVW** are computed with the full IAU 2006/2000A Earth orientation (ERFA;
precession, nutation, UT1, polar motion) and agree with casacore's `measures`
to <0.1 m. MS convention: `baseline = ant2 − ant1`. Phase:
$\exp (2\pi i(ul + vm + w(n-1))\nu/c)$. No network access is needed (bundled IERS
data; for epochs past it a degraded-accuracy prediction is used).

## Current limitations

* Point sources only; no primary beam or direction-dependent gains yet
  (a natural hook: multiply `coh` in `predict.source_coherency`).
* Fringe (phase) smearing within a dump is not modelled – only flux
  variation is averaged. Matters for off-axis sources on long baselines
  with long dumps.
* Single SPW / single field when creating a new MS (injection selects
  `field_id`/`ddid`).
* No scattering/scintillation, no cross-polar leakage (D-terms).
