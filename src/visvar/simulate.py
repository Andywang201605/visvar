"""High-level entry points: :func:`simulate` and :func:`inject`."""
from __future__ import annotations

import numpy as np
from astropy.time import Time
from casacore.tables import table, makecoldesc

from .array import antenna_uvw
from .msio import append_rows, create_empty_ms, read_ms_info
from .observation import Observation
from .predict import add_noise, predict_visibilities, thermal_sigma
from .sky import Sky, Source


def _auto_chunk_rows(nchan, ncorr, max_elems=1.5e7):
    """Rows per chunk so that one (nrow, nchan, ncorr) complex array is ~240 MB."""
    return int(max(1, max_elems // (nchan * ncorr)))


def _as_sky(sky):
    if isinstance(sky, Sky):
        return sky
    if isinstance(sky, Source):
        return Sky([sky])
    return Sky(list(sky))


def simulate(
        sky, obs: Observation, out, sefd=None, seed=None,
        chunk_rows=None, nt_sub=None, nf_sub=None, overwrite=True,
        verbose=False
    ):
    """Create a new Measurement Set containing the simulated sky.

    Parameters
    ----------
    sky : Sky, Source or list of Source
    obs : Observation
    out : str   output MS path
    sefd : float, optional
        System equivalent flux density (Jy).  If given, thermal noise is added
        with per-component rms ``sefd / sqrt(2 * chan_width * dt)``.
    nt_sub, nf_sub : int, optional
        Force the number of time / frequency sub-samples used when averaging
        lightcurves over each integration / channel.  Default: automatic.
    """
    sky = _as_sky(sky)
    rng = np.random.default_rng(seed)
    corrs = list(obs.corrs)
    a1, a2 = obs.baselines()
    nbl = a1.size
    ra0 = obs.phase_centre.icrs.ra.rad
    dec0 = obs.phase_centre.icrs.dec.rad
    times_all = obs.times_mjd_s()
    t_ref = obs.t0_mjd_s

    create_empty_ms(out, obs, overwrite=overwrite)
    sigma = None
    if sefd is not None:
        sigma = thermal_sigma(sefd, obs.chan_width, obs.dt)

    chunk_rows = chunk_rows or _auto_chunk_rows(obs.nchan, len(corrs))
    nt_chunk = max(1, chunk_rows // nbl)
    for i0 in range(0, times_all.size, nt_chunk):
        times = times_all[i0:i0 + nt_chunk]
        nt = times.size
        auvw = antenna_uvw(obs.array.positions, times, ra0, dec0)  # (nt, nant, 3)
        uvw = (auvw[:, a2] - auvw[:, a1]).reshape(nt * nbl, 3)
        tidx = np.repeat(np.arange(nt), nbl)
        vis = predict_visibilities(
            sky, obs.phase_centre, times - t_ref, obs.dt, tidx,
            uvw, obs.freqs, obs.chan_width, corrs, nt_sub, nf_sub
        )
        if sigma is not None:
            vis = add_noise(vis, sigma, rng)
        append_rows(
            out, np.repeat(times, nbl), obs.dt, np.tile(a1, nt), np.tile(a2, nt),
            uvw, vis, sigma=None if sigma is None else np.full(len(corrs), sigma)
        )
        if verbose:
            print(f"wrote integrations {i0}-{i0 + nt} / {times_all.size}")
    return out


def inject(
        ms, sky, column="DATA", mode="add", sefd=None, seed=None,
        field_id=0, ddid=0, spw_id=0, t_ref=None, chunk_rows=None,
        nt_sub=None, nf_sub=None, verbose=False
    ):
    """Inject the sky into an existing Measurement Set (in place).

    Uses the MS's own UVW, TIME, INTERVAL, antennas, channels and phase
    centre.  Lightcurve time zero is ``t_ref`` (MJD UTC seconds or astropy
    ``Time``); by default the start of the first integration.

    Parameters
    ----------
    ms : str   measurement set to inject into
    sky : SkyModel   sky model to inject
    column : str   column to write (created from DATA's description if absent)
    mode : {"add", "replace"}
    sefd : float, optional  also add thermal noise (see :func:`simulate`)
    """
    if mode not in ("add", "replace"):
        raise ValueError("mode must be 'add' or 'replace'")
    sky = _as_sky(sky)
    rng = np.random.default_rng(seed)
    info = read_ms_info(ms, field_id=field_id, spw_id=spw_id)
    freqs, cw, pc, corrs = info["freqs"], info["chan_width"], info["phase_centre"], info["corrs"]

    with table(ms, readonly=False, ack=False) as t:
        if column not in t.colnames():
            desc = makecoldesc(column, t.getcoldesc("DATA"))
            t.addcols(desc)
            # initialise to zero so "add" is well defined
            zeros_init = True
        else:
            zeros_init = False
        n = t.nrows()
        chunk_rows = chunk_rows or _auto_chunk_rows(len(freqs), len(corrs))
        if t_ref is None:
            tt, ii = t.getcol("TIME"), t.getcol("INTERVAL")
            t_ref = float(np.min(tt - ii / 2.0))
        elif isinstance(t_ref, Time):
            t_ref = t_ref.utc.mjd * 86400.0

        for r0 in range(0, n, chunk_rows):
            nr = min(chunk_rows, n - r0)
            tim = t.getcol("TIME", r0, nr)
            ivl = t.getcol("INTERVAL", r0, nr)
            dd = t.getcol("DATA_DESC_ID", r0, nr)
            fld = t.getcol("FIELD_ID", r0, nr)
            sel = (dd == ddid) & (fld == field_id)
            if zeros_init or mode == "replace":
                cur = np.zeros_like(t.getcol("DATA", r0, nr))
            else:
                cur = t.getcol(column, r0, nr)
            if sel.any():
                uvw = t.getcol("UVW", r0, nr)[sel]
                ut, inv = np.unique(tim[sel], return_index=False, return_inverse=True)
                # interval per unique time (robust, vectorised)
                dt_u = np.empty(ut.size)
                dt_u[inv] = ivl[sel]
                vis = predict_visibilities(
                    sky, pc, ut - t_ref, dt_u, inv, uvw, freqs, cw,
                    corrs, nt_sub, nf_sub
                )
                if sefd is not None:
                    sig = thermal_sigma(sefd, cw, ivl[sel])[:, None, None]
                    vis = add_noise(vis, sig, rng)
                cur[sel] += vis.astype(cur.dtype)
            t.putcol(column, cur, r0, nr)
            if verbose:
                print(f"rows {r0}-{r0 + nr} / {n}")
    return ms
