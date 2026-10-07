"""Observation set-up (what to simulate)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

from .array import Array


def _utc_mjd_seconds(t: Time):
    return t.utc.mjd * 86400.0


@dataclass
class Observation:
    """Parameters of a simulated observation.

    ``start`` is the start of the first integration; integration ``i`` is
    centred at ``start + (i + 0.5) * dt``.
    """

    array: Array
    phase_centre: SkyCoord
    start: Time
    duration: float          # s
    dt: float                # integration time, s
    freqs: np.ndarray        # channel centres, Hz
    chan_width: Optional[float] = None  # Hz
    corrs: Sequence[str] = ("XX", "XY", "YX", "YY")
    autocorr: bool = False
    field_name: str = "FIELD0"
    telescope_name: Optional[str] = None
    observer: str = "dummy"

    def __post_init__(self):
        self.freqs = np.atleast_1d(np.asarray(self.freqs, float))
        if self.chan_width is None:
            self.chan_width = (float(np.median(np.diff(self.freqs)))
                               if self.freqs.size > 1 else 1e6)
        if not isinstance(self.start, Time):
            raise TypeError("start must be an astropy Time")

    @property
    def ntime(self):
        return int(round(self.duration / self.dt))

    @property
    def nchan(self):
        return self.freqs.size

    @property
    def t0_mjd_s(self):
        """MJD(UTC) seconds of the observation start (lightcurve t = 0)."""
        return _utc_mjd_seconds(self.start)

    def times_mjd_s(self):
        """Integration centres in MJD(UTC) seconds (as stored in the MS)."""
        return self.t0_mjd_s + (np.arange(self.ntime) + 0.5) * self.dt

    def baselines(self):
        n = self.array.nant
        k = 0 if self.autocorr else 1
        a1, a2 = np.triu_indices(n, k=k)
        return a1.astype(np.int32), a2.astype(np.int32)
