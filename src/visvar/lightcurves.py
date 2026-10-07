"""Lightcurve models.

A lightcurve maps (time, frequency) -> flux density (Jy).  Times are seconds
since the observation reference time (``t_ref``, normally the start of the
observation), which keeps float64 phase precision high for fast pulsars.

All lightcurves implement :meth:`Lightcurve.evaluate`.  The simulator never
samples at the integration midpoint only: it calls :meth:`Lightcurve.mean`,
which averages over each integration (and channel) with automatic
sub-sampling, so pulses shorter than the dump time are handled correctly.
"""
from __future__ import annotations

import numpy as np

# Dispersion constant: delay = K_DM * DM * (nu_GHz**-2 - nu_ref_GHz**-2)  [s]
K_DM = 4.148808e-3  # s GHz^2 cm^3 / pc

MAX_SUB = 512


class Lightcurve:
    """Base class.  Subclasses implement :meth:`evaluate`."""

    #: shortest characteristic timescale (s); used to choose sub-sampling
    timescale = np.inf

    def evaluate(self, t, freq):
        """Flux density (Jy).

        Parameters
        ----------
        t : array (...,)   seconds since reference time
        freq : array (nchan,)   Hz

        Returns
        -------
        array (..., nchan)
        """
        raise NotImplementedError

    # -- sub-sampling helpers -------------------------------------------------
    def n_time_sub(self, dt):
        if not np.isfinite(self.timescale):
            return 1
        return int(np.clip(np.ceil(4.0 * dt / self.timescale), 1, MAX_SUB))

    def n_freq_sub(self, freq, df):
        return 1

    def mean(self, t, dt, freq, df, nt_sub=None, nf_sub=None):
        """Average flux over integrations ``[t-dt/2, t+dt/2]`` and channels
        ``[f-df/2, f+df/2]``.  Returns array (ntime, nchan)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        freq = np.atleast_1d(np.asarray(freq, dtype=float))
        dt = float(dt)
        df = float(df)
        nt = nt_sub or self.n_time_sub(dt)
        nf = nf_sub or self.n_freq_sub(freq, df)
        toff = ((np.arange(nt) + 0.5) / nt - 0.5) * dt
        foff = ((np.arange(nf) + 0.5) / nf - 0.5) * df
        out = np.zeros((t.size, freq.size))
        for k in range(nf):
            fk = freq + foff[k]
            for j in range(nt):
                out += self.evaluate(t + toff[j], fk)
        return out / (nt * nf)

    # -- arithmetic ---------------------------------------------------------
    def __add__(self, other):
        return _Sum(self, as_lightcurve(other))

    __radd__ = __add__

    def __mul__(self, scale):
        return _Scaled(self, float(scale))

    __rmul__ = __mul__


def as_lightcurve(x):
    if x is None:
        return None
    if isinstance(x, Lightcurve):
        return x
    if callable(x):
        return FromCallable(x)
    return Constant(float(x))


class _Sum(Lightcurve):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.timescale = min(a.timescale, b.timescale)

    def evaluate(self, t, freq):
        return self.a.evaluate(t, freq) + self.b.evaluate(t, freq)

    def n_freq_sub(self, freq, df):
        return max(self.a.n_freq_sub(freq, df), self.b.n_freq_sub(freq, df))


class _Scaled(Lightcurve):
    def __init__(self, a, s):
        self.a, self.s = a, s
        self.timescale = a.timescale

    def evaluate(self, t, freq):
        return self.s * self.a.evaluate(t, freq)

    def n_freq_sub(self, freq, df):
        return self.a.n_freq_sub(freq, df)


class Constant(Lightcurve):
    """Steady source with flux ``flux`` (Jy)."""

    def __init__(self, flux):
        self.flux = float(flux)

    def evaluate(self, t, freq):
        t = np.asarray(t, float)
        return np.full(t.shape + (np.size(freq),), self.flux)


class Sinusoid(Lightcurve):
    """``mean * (1 + amplitude * sin(2 pi (t/period + phase0)))``.

    ``amplitude`` is the fractional modulation depth (1.0 = 100 %).
    """

    def __init__(self, period, mean=1.0, amplitude=0.5, phase0=0.0):
        self.period = float(period)
        self.mean_flux = float(mean)
        self.amplitude = float(amplitude)
        self.phase0 = float(phase0)
        self.timescale = self.period / 4.0

    def evaluate(self, t, freq):
        t = np.asarray(t, float)
        s = 1.0 + self.amplitude * np.sin(2 * np.pi * (t / self.period + self.phase0))
        return (self.mean_flux * s)[..., None] * np.ones(np.size(freq))

    def mean(self, t, dt, freq, df, nt_sub=None, nf_sub=None):
        # Exact boxcar average of a sinusoid (no sub-sampling needed).
        t = np.atleast_1d(np.asarray(t, float))
        freq = np.atleast_1d(freq)
        w = 2 * np.pi / self.period
        sinc = np.sinc(dt / self.period)  # sin(w dt/2)/(w dt/2)
        s = 1.0 + self.amplitude * sinc * np.sin(w * t + 2 * np.pi * self.phase0)
        return (self.mean_flux * s)[:, None] * np.ones(freq.size)


class PulseTrain(Lightcurve):
    """Periodic pulsed source (pulsar-like).

    Parameters
    ----------
    period : float
        Spin period at ``t = 0`` (s).
    peak : float
        Peak flux density of the pulse (Jy), *before* adding ``baseline``.
        Ignored if ``mean_flux`` is given.
    mean_flux : float, optional
        If set, the profile is normalised so that the period-averaged flux
        is ``mean_flux`` (Jy).  (Often what pulsar catalogues quote.)
    width : float
        Profile width as a fraction of the period (FWHM for ``gaussian`` /
        ``vonmises``, full width for ``boxcar``).
    profile : {"gaussian", "vonmises", "boxcar"} or callable
        Pulse shape.  A callable takes pulse phase in [-0.5, 0.5) and returns
        values with a maximum of ~1.
    pdot : float
        Period derivative (s/s).  Phase = t/P - 0.5 pdot t^2 / P^2.
    phase0 : float
        Pulse phase (cycles) at t = 0 (the pulse peak is at phase 0).
    dm : float
        Dispersion measure (pc cm^-3).  Delays are relative to ``ref_freq``.
    ref_freq : float
        Frequency (Hz) at which the pulse arrives at ``phase0``.  Defaults to
        infinite frequency if ``None`` (i.e. delay = K dm / nu^2).
    baseline : float
        Constant off-pulse flux (Jy).
    """

    def __init__(self, period, peak=1.0, width=0.05, profile="gaussian",
                 pdot=0.0, phase0=0.0, dm=0.0, ref_freq=None, baseline=0.0,
                 mean_flux=None):
        self.period = float(period)
        self.width = float(width)
        self.pdot = float(pdot)
        self.phase0 = float(phase0)
        self.dm = float(dm)
        self.ref_freq = None if ref_freq is None else float(ref_freq)
        self.baseline = float(baseline)
        self.profile = profile
        self._shape = self._make_profile(profile)
        if mean_flux is not None:
            ph = (np.arange(4096) + 0.5) / 4096 - 0.5
            avg = float(np.mean(self._shape(ph)))
            peak = (float(mean_flux) - self.baseline) / avg
        self.peak = float(peak)
        self.timescale = max(self.width * self.period, 1e-12)

    def _make_profile(self, profile):
        w = self.width
        if callable(profile):
            return profile
        if profile == "gaussian":
            sig = w / (2 * np.sqrt(2 * np.log(2)))
            return lambda ph: np.exp(-0.5 * (ph / sig) ** 2)
        if profile == "vonmises":
            # FWHM of exp(kappa (cos x - 1)), x = 2 pi ph
            # cos(pi w) = 1 - ln2/kappa
            kappa = np.log(2) / (1 - np.cos(np.pi * w))
            return lambda ph: np.exp(kappa * (np.cos(2 * np.pi * ph) - 1))
        if profile == "boxcar":
            return lambda ph: (np.abs(ph) <= w / 2).astype(float)
        raise ValueError(f"unknown profile {profile!r}")

    def delay(self, freq):
        """Dispersive delay (s) at ``freq`` (Hz) relative to ``ref_freq``."""
        if self.dm == 0.0:
            return np.zeros(np.shape(freq))
        g = np.asarray(freq, float) / 1e9
        d = K_DM * self.dm * g ** -2
        if self.ref_freq is not None:
            d = d - K_DM * self.dm * (self.ref_freq / 1e9) ** -2
        return d

    def pulse_phase(self, t):
        t = np.asarray(t, float)
        return t / self.period - 0.5 * self.pdot * t ** 2 / self.period ** 2 + self.phase0

    def evaluate(self, t, freq):
        t = np.asarray(t, float)[..., None]
        freq = np.atleast_1d(np.asarray(freq, float))
        te = t - self.delay(freq)  # (..., nchan)
        ph = self.pulse_phase(te)
        ph = ph - np.round(ph)  # wrap to [-0.5, 0.5]
        return self.baseline + self.peak * self._shape(ph)

    def n_freq_sub(self, freq, df):
        if self.dm == 0.0:
            return 1
        f = np.asarray(freq, float)
        smear = np.max(np.abs(self.delay(f - df / 2) - self.delay(f + df / 2)))
        return int(np.clip(np.ceil(4.0 * smear / self.timescale), 1, 64))

    # -- exact integration over each dump ----------------------------------
    def _cumulative_table(self):
        if getattr(self, "_cum", None) is None:
            n = int(np.clip(np.ceil(64.0 / max(self.width, 1e-6)), 4096, 1 << 18))
            mid = (np.arange(n) + 0.5) / n - 0.5
            vals = np.asarray(self._shape(mid), float)
            cum = np.concatenate([[0.0], np.cumsum(vals) / n])
            self._cum = (np.linspace(0.0, 1.0, n + 1), cum, cum[-1])
        return self._cum

    def _cum_at(self, ph):
        """Integral of the profile over phase from -0.5 to ``ph`` (unwrapped)."""
        grid, cum, total = self._cumulative_table()
        x = ph + 0.5
        k = np.floor(x)
        return k * total + np.interp(x - k, grid, cum)

    def mean(self, t, dt, freq, df, nt_sub=None, nf_sub=None):
        """Exact (table-based) average over each integration, so pulses much
        shorter than the dump -- even many periods per dump -- are handled
        correctly with no time sub-sampling.  Channel smearing from dispersion
        is handled by sub-dividing each channel."""
        if nt_sub is not None:  # explicit brute-force request
            return super().mean(t, dt, freq, df, nt_sub, nf_sub)
        t = np.atleast_1d(np.asarray(t, float))
        freq = np.atleast_1d(np.asarray(freq, float))
        nf = nf_sub or self.n_freq_sub(freq, df)
        foff = ((np.arange(nf) + 0.5) / nf - 0.5) * df
        out = np.zeros((t.size, freq.size))
        ta, tb = (t - 0.5 * dt)[:, None], (t + 0.5 * dt)[:, None]
        for k in range(nf):
            d = self.delay(freq + foff[k])[None, :]
            pa = self.pulse_phase(ta - d)
            pb = self.pulse_phase(tb - d)
            dph = pb - pa
            with np.errstate(divide="ignore", invalid="ignore"):
                avg = (self._cum_at(pb) - self._cum_at(pa)) / dph
            tiny = np.abs(dph) < 1e-12
            if tiny.any():
                avg = np.where(tiny, self._shape(pa - np.round(pa)), avg)
            out += self.baseline + self.peak * avg
        return out / nf


class Tabulated(Lightcurve):
    """Lightcurve from samples ``(times, flux)`` with linear interpolation.

    If ``period`` is given the table is treated as one cycle and repeated.
    Flux is independent of frequency.
    """

    def __init__(self, times, flux, period=None, timescale=None):
        order = np.argsort(times)
        self.times = np.asarray(times, float)[order]
        self.flux = np.asarray(flux, float)[order]
        self.period = period
        if timescale is None:
            d = np.diff(self.times)
            timescale = float(np.min(d)) if d.size else np.inf
        self.timescale = timescale

    def evaluate(self, t, freq):
        t = np.asarray(t, float)
        if self.period is not None:
            t = self.times[0] + np.mod(t - self.times[0], self.period)
        v = np.interp(t, self.times, self.flux, left=0.0, right=0.0)
        return v[..., None] * np.ones(np.size(freq))


class Burst(Lightcurve):
    """Fast-rise exponential-decay transient, optionally dispersed.

    flux = peak * exp(-(t - t0)/tau) for t >= t0 (Gaussian-smoothed rise of
    width ``rise``).
    """

    def __init__(self, t0, peak=1.0, tau=1.0, rise=None, dm=0.0, ref_freq=None):
        self.t0, self.peak, self.tau = float(t0), float(peak), float(tau)
        self.rise = float(rise) if rise else self.tau / 10.0
        self.dm = float(dm)
        self.ref_freq = ref_freq
        self.timescale = self.rise

    def delay(self, freq):
        if self.dm == 0.0:
            return np.zeros(np.shape(freq))
        d = K_DM * self.dm * (np.asarray(freq, float) / 1e9) ** -2
        if self.ref_freq is not None:
            d = d - K_DM * self.dm * (self.ref_freq / 1e9) ** -2
        return d

    def evaluate(self, t, freq):
        t = np.asarray(t, float)[..., None]
        x = t - self.t0 - self.delay(np.atleast_1d(freq))
        # exponential decay convolved-ish with gaussian rise: use smooth step
        step = 0.5 * (1 + np.tanh(x / self.rise))
        return self.peak * step * np.exp(-np.clip(x, 0, None) / self.tau)

    def n_freq_sub(self, freq, df):
        if self.dm == 0.0:
            return 1
        f = np.asarray(freq, float)
        smear = np.max(np.abs(self.delay(f - df / 2) - self.delay(f + df / 2)))
        return int(np.clip(np.ceil(4.0 * smear / self.timescale), 1, 64))


class FromCallable(Lightcurve):
    """Wrap ``func(t, freq) -> flux`` (or ``func(t) -> flux``).

    ``func`` receives ``t`` with shape (..., 1) and ``freq`` (nchan,), so
    broadcasting gives (..., nchan).  Give ``timescale`` (s) to enable
    automatic sub-sampling within integrations.
    """

    def __init__(self, func, timescale=np.inf):
        self.func = func
        self.timescale = timescale
        try:
            import inspect
            self._nargs = len(inspect.signature(func).parameters)
        except (TypeError, ValueError):
            self._nargs = 2

    def evaluate(self, t, freq):
        t = np.asarray(t, float)[..., None]
        freq = np.atleast_1d(np.asarray(freq, float))
        v = self.func(t) if self._nargs == 1 else self.func(t, freq)
        return np.broadcast_to(v, t.shape[:-1] + (freq.size,)).copy()
