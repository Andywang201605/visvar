"""Visibility prediction for time-variable point sources."""
from __future__ import annotations

import numpy as np

from .polarisation import stokes_to_corr

C = 299792458.0


def source_coherency(source, t_rel, dt, freqs, chan_width, corrs,
                     nt_sub=None, nf_sub=None):
    """Integration/channel-averaged coherency of ``source``.

    Parameters
    ----------
    t_rel : (ntime,) integration centres, seconds since lightcurve t = 0
    dt : float or (ntime,) integration lengths (s)

    Returns
    -------
    (ntime, nchan, ncorr) complex array
    """
    t_rel = np.atleast_1d(np.asarray(t_rel, float))
    dts = np.broadcast_to(np.asarray(dt, float), t_rel.shape)
    ntime, nchan = t_rel.size, np.size(freqs)

    stokes = {}
    for key, lc in source.stokes().items():
        if lc is None:
            stokes[key] = np.zeros((ntime, nchan))
            continue
        out = np.empty((ntime, nchan))
        # group by distinct integration length (normally just one)
        for d in np.unique(dts):
            sel = dts == d
            out[sel] = lc.mean(t_rel[sel], d, freqs, chan_width, nt_sub, nf_sub)
        stokes[key] = out * source.spectral_scale(freqs)[None, :]

    coh = np.empty((ntime, nchan, len(corrs)), dtype=np.complex128)
    for i, c in enumerate(corrs):
        coh[:, :, i] = stokes_to_corr(c, stokes["I"], stokes["Q"], stokes["U"], stokes["V"])
    return coh


def phase_term(uvw, lmn, freqs):
    """exp(2 pi i (u l + v m + w (n-1)) nu / c) for rows (nrow,) -> (nrow, nchan)."""
    l, m, n = lmn
    path = uvw[:, 0] * l + uvw[:, 1] * m + uvw[:, 2] * (n - 1.0)  # metres
    return np.exp(2j * np.pi * path[:, None] * (np.asarray(freqs)[None, :] / C))


def predict_visibilities(sky, phase_centre, t_rel, dt, time_index, uvw, freqs,
                         chan_width, corrs, nt_sub=None, nf_sub=None):
    """Predict visibilities for rows of a measurement set.

    Parameters
    ----------
    t_rel : (ntime_unique,) unique integration centres (s since lightcurve t=0)
    dt : float or (ntime_unique,) integration time
    time_index : (nrow,) index into ``t_rel`` for each row
    uvw : (nrow, 3) metres (MS convention)

    Returns
    -------
    (nrow, nchan, ncorr) complex128
    """
    nrow = uvw.shape[0]
    vis = np.zeros((nrow, np.size(freqs), len(corrs)), dtype=np.complex128)
    for src in sky:
        coh = source_coherency(src, t_rel, dt, freqs, chan_width, corrs, nt_sub, nf_sub)
        ph = phase_term(uvw, src.lmn(phase_centre), freqs)
        vis += coh[time_index] * ph[:, :, None]
    return vis


def thermal_sigma(sefd, chan_width, dt):
    """Per-component (real or imaginary) noise rms (Jy) for one correlation."""
    return sefd / np.sqrt(2.0 * chan_width * dt)


def add_noise(vis, sigma, rng):
    sigma = np.asarray(sigma, float)
    noise = rng.normal(size=vis.shape) + 1j * rng.normal(size=vis.shape)
    return vis + noise * sigma
