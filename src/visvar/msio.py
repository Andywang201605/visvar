"""Measurement Set creation and I/O (python-casacore)."""
from __future__ import annotations

import os
import shutil

import numpy as np
from casacore.tables import (default_ms, default_ms_subtable, makearrcoldesc,
                             maketabdesc, table, makescacoldesc)

from .polarisation import CORR_CODES, corr_product, corr_to_feeds

MS_SUBTABLES = ["ANTENNA", "DATA_DESCRIPTION", "FEED", "FIELD", "FLAG_CMD",
                "HISTORY", "OBSERVATION", "POINTING", "POLARIZATION",
                "PROCESSOR", "SOURCE", "SPECTRAL_WINDOW", "STATE"]


def _fill_subtable(ms_path, name, rows: dict, nrows=1):
    """Create ``ms_path/name`` with default columns and add ``nrows`` rows."""
    sub_path = os.path.join(ms_path, name)
    st = default_ms_subtable(name, sub_path)
    if nrows:
        st.addrows(nrows)
        for col, val in rows.items():
            st.putcol(col, val)
    st.flush()
    st.close()
    with table(ms_path, readonly=False, ack=False) as t:
        t.putkeyword(name, f"Table: {sub_path}")


def create_empty_ms(path, obs, overwrite=True, extra_columns=()):
    """Create an MS with all metadata for ``obs`` and an empty main table."""
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)

    nchan, ncorr = obs.nchan, len(obs.corrs)
    cols = [
        makearrcoldesc("DATA", 0j, ndim=2, valuetype="complex"),
        makearrcoldesc("FLAG", False, ndim=2, valuetype="boolean"),
    ]
    for c in extra_columns:
        cols.append(makearrcoldesc(c, 0j, ndim=2, valuetype="complex"))
    tabdesc = maketabdesc(cols)
    ms = default_ms(path, tabdesc)
    ms.close()

    arr = obs.array
    ra0 = obs.phase_centre.icrs.ra.rad
    dec0 = obs.phase_centre.icrs.dec.rad
    t0 = obs.t0_mjd_s
    t1 = t0 + obs.duration
    feeds = corr_to_feeds(obs.corrs)

    # ANTENNA
    _fill_subtable(path, "ANTENNA", {
        "NAME": np.array(arr.names),
        "STATION": np.array([arr.name] * arr.nant),
        "TYPE": np.array(["GROUND-BASED"] * arr.nant),
        "MOUNT": np.array([arr.mount] * arr.nant),
        "POSITION": arr.positions,
        "OFFSET": np.zeros((arr.nant, 3)),
        "DISH_DIAMETER": np.full(arr.nant, arr.diameter),
        "FLAG_ROW": np.zeros(arr.nant, bool),
    }, nrows=arr.nant)

    # SPECTRAL_WINDOW
    f = obs.freqs
    _fill_subtable(path, "SPECTRAL_WINDOW", {
        "NUM_CHAN": np.array([nchan], np.int32),
        "NAME": np.array(["SPW0"]),
        "REF_FREQUENCY": np.array([f[0]]),
        "CHAN_FREQ": f[None, :],
        "CHAN_WIDTH": np.full((1, nchan), obs.chan_width),
        "EFFECTIVE_BW": np.full((1, nchan), obs.chan_width),
        "RESOLUTION": np.full((1, nchan), obs.chan_width),
        "MEAS_FREQ_REF": np.array([5], np.int32),  # TOPO
        "TOTAL_BANDWIDTH": np.array([obs.chan_width * nchan]),
        "NET_SIDEBAND": np.array([1], np.int32),
        "IF_CONV_CHAIN": np.array([0], np.int32),
        "FREQ_GROUP": np.array([0], np.int32),
        "FREQ_GROUP_NAME": np.array(["Group 1"]),
        "FLAG_ROW": np.array([False]),
    })

    # POLARIZATION
    _fill_subtable(path, "POLARIZATION", {
        "NUM_CORR": np.array([ncorr], np.int32),
        "CORR_TYPE": np.array([[CORR_CODES[c] for c in obs.corrs]], np.int32),
        "CORR_PRODUCT": corr_product(obs.corrs)[None, :, :],
        "FLAG_ROW": np.array([False]),
    })

    # DATA_DESCRIPTION
    _fill_subtable(path, "DATA_DESCRIPTION", {
        "SPECTRAL_WINDOW_ID": np.array([0], np.int32),
        "POLARIZATION_ID": np.array([0], np.int32),
        "FLAG_ROW": np.array([False]),
    })

    # FIELD
    pd = np.array([[[ra0, dec0]]])
    _fill_subtable(path, "FIELD", {
        "NAME": np.array([obs.field_name]),
        "NUM_POLY": np.array([0], np.int32),
        "DELAY_DIR": pd, "PHASE_DIR": pd, "REFERENCE_DIR": pd,
        "SOURCE_ID": np.array([0], np.int32),
        "TIME": np.array([t0]),
        "FLAG_ROW": np.array([False]),
    })

    # OBSERVATION
    _fill_subtable(path, "OBSERVATION", {
        "TELESCOPE_NAME": np.array([obs.telescope_name or arr.name]),
        "OBSERVER": np.array([obs.observer]),
        "TIME_RANGE": np.array([[t0, t1]]),
        "SCHEDULE_TYPE": np.array([""]),
        "PROJECT": np.array(["test"]),
        "RELEASE_DATE": np.array([t0]),
        "FLAG_ROW": np.array([False]),
    })

    # FEED (one per antenna)
    nrec = len(feeds)
    _fill_subtable(path, "FEED", {
        "ANTENNA_ID": np.arange(arr.nant, dtype=np.int32),
        "FEED_ID": np.zeros(arr.nant, np.int32),
        "SPECTRAL_WINDOW_ID": np.full(arr.nant, -1, np.int32),
        "TIME": np.zeros(arr.nant),
        "INTERVAL": np.zeros(arr.nant),
        "NUM_RECEPTORS": np.full(arr.nant, nrec, np.int32),
        "BEAM_ID": np.full(arr.nant, -1, np.int32),
        "BEAM_OFFSET": np.zeros((arr.nant, 2, nrec)),
        "POLARIZATION_TYPE": np.tile(np.array(feeds), (arr.nant, 1)),
        "POL_RESPONSE": np.tile(np.eye(nrec, dtype=complex), (arr.nant, 1, 1)),
        "RECEPTOR_ANGLE": np.zeros((arr.nant, nrec)),
    }, nrows=arr.nant)

    # PROCESSOR / STATE: one default row each (some CASA tasks expect them)
    _fill_subtable(path, "PROCESSOR", {
        "TYPE": np.array(["CORRELATOR"]), "SUB_TYPE": np.array([""]),
        "TYPE_ID": np.array([0], np.int32), "MODE_ID": np.array([0], np.int32),
        "FLAG_ROW": np.array([False])})
    _fill_subtable(path, "STATE", {
        "SIG": np.array([True]), "REF": np.array([False]),
        "CAL": np.array([0.0]), "LOAD": np.array([0.0]),
        "SUB_SCAN": np.array([0], np.int32), "OBS_MODE": np.array(["OBSERVE_TARGET"]),
        "FLAG_ROW": np.array([False])})
    for name in ("FLAG_CMD", "HISTORY", "POINTING", "SOURCE"):
        _fill_subtable(path, name, {}, nrows=0)
    return path


def append_rows(path, time, interval, ant1, ant2, uvw, data, flag=None,
                sigma=None, field_id=0, scan=1, extra=None):
    """Append rows to the main table.

    ``data`` has shape (nrow, nchan, ncorr)."""
    nrow = len(time)
    with table(path, readonly=False, ack=False) as t:
        start = t.nrows()
        t.addrows(nrow)
        sl = dict(startrow=start, nrow=nrow)
        t.putcol("TIME", np.asarray(time, float), **sl)
        t.putcol("TIME_CENTROID", np.asarray(time, float), **sl)
        t.putcol("INTERVAL", np.full(nrow, interval), **sl)
        t.putcol("EXPOSURE", np.full(nrow, interval), **sl)
        t.putcol("ANTENNA1", np.asarray(ant1, np.int32), **sl)
        t.putcol("ANTENNA2", np.asarray(ant2, np.int32), **sl)
        t.putcol("FEED1", np.zeros(nrow, np.int32), **sl)
        t.putcol("FEED2", np.zeros(nrow, np.int32), **sl)
        t.putcol("DATA_DESC_ID", np.zeros(nrow, np.int32), **sl)
        t.putcol("PROCESSOR_ID", np.zeros(nrow, np.int32), **sl)
        t.putcol("FIELD_ID", np.full(nrow, field_id, np.int32), **sl)
        t.putcol("SCAN_NUMBER", np.full(nrow, scan, np.int32), **sl)
        t.putcol("ARRAY_ID", np.zeros(nrow, np.int32), **sl)
        t.putcol("OBSERVATION_ID", np.zeros(nrow, np.int32), **sl)
        t.putcol("STATE_ID", np.zeros(nrow, np.int32), **sl)
        t.putcol("UVW", np.asarray(uvw, float), **sl)
        t.putcol("DATA", data.astype(np.complex64), **sl)
        t.putcol("FLAG", np.zeros(data.shape, bool) if flag is None else flag, **sl)
        t.putcol("FLAG_ROW", np.zeros(nrow, bool), **sl)
        ncorr = data.shape[2]
        sig = np.ones((nrow, ncorr)) if sigma is None else np.broadcast_to(sigma, (nrow, ncorr))
        t.putcol("SIGMA", sig.astype(np.float32), **sl)
        t.putcol("WEIGHT", (1.0 / np.asarray(sig) ** 2).astype(np.float32), **sl)
        for col, val in (extra or {}).items():
            t.putcol(col, val, **sl)


def read_ms_info(path, field_id=0, spw_id=0):
    """Metadata needed to inject sources into an existing MS."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    with table(f"{path}::SPECTRAL_WINDOW", ack=False) as t:
        freqs = t.getcol("CHAN_FREQ")[spw_id]
        cw = float(np.mean(np.abs(t.getcol("CHAN_WIDTH")[spw_id])))
    with table(f"{path}::FIELD", ack=False) as t:
        pd = t.getcol("PHASE_DIR")[field_id, 0]
    with table(f"{path}::POLARIZATION", ack=False) as t:
        codes = t.getcol("CORR_TYPE")[0]
    inv = {v: k for k, v in CORR_CODES.items()}
    corrs = [inv[int(c)] for c in codes]
    return dict(freqs=freqs, chan_width=cw,
                phase_centre=SkyCoord(pd[0] * u.rad, pd[1] * u.rad, frame="icrs"),
                corrs=corrs)
