"""Sky model: sources with Stokes lightcurves."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from .lightcurves import Lightcurve, as_lightcurve


@dataclass
class Source:
    """A point source.

    Parameters
    ----------
    coord : SkyCoord or (ra_deg, dec_deg)
    I, Q, U, V : Lightcurve, float, callable or None
        Stokes lightcurves.  Floats become constant sources; callables
        ``f(t, freq)`` are wrapped.  Unset polarisations are zero.
    spectral_index, ref_freq
        Static power-law ``(nu/ref_freq)**spectral_index`` applied on top of
        every Stokes lightcurve.
    """

    coord: SkyCoord
    I: Lightcurve = 1.0
    Q: Optional[Lightcurve] = None
    U: Optional[Lightcurve] = None
    V: Optional[Lightcurve] = None
    spectral_index: float = 0.0
    ref_freq: float = 1.0e9
    name: str = "source"

    def __post_init__(self):
        if not isinstance(self.coord, SkyCoord):
            ra, dec = self.coord
            self.coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        self.I = as_lightcurve(self.I)
        self.Q = as_lightcurve(self.Q)
        self.U = as_lightcurve(self.U)
        self.V = as_lightcurve(self.V)

    @property
    def ra(self):
        return self.coord.icrs.ra.rad

    @property
    def dec(self):
        return self.coord.icrs.dec.rad

    def stokes(self):
        return {"I": self.I, "Q": self.Q, "U": self.U, "V": self.V}

    def spectral_scale(self, freq):
        if self.spectral_index == 0.0:
            return np.ones(np.size(freq))
        return (np.asarray(freq, float) / self.ref_freq) ** self.spectral_index

    def lmn(self, phase_centre: SkyCoord):
        ra0, dec0 = phase_centre.icrs.ra.rad, phase_centre.icrs.dec.rad
        return radec_to_lmn(self.ra, self.dec, ra0, dec0)


def radec_to_lmn(ra, dec, ra0, dec0):
    dra = ra - ra0
    l = np.cos(dec) * np.sin(dra)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)
    n = np.sqrt(np.clip(1.0 - l * l - m * m, 0.0, None))
    return l, m, n


class Sky:
    """A collection of :class:`Source`.  Static field sources are just
    sources with constant lightcurves."""

    def __init__(self, sources: Sequence[Source] = ()):
        self.sources = list(sources)

    def add(self, source: Source):
        self.sources.append(source)
        return self

    def __iter__(self):
        return iter(self.sources)

    def __len__(self):
        return len(self.sources)
