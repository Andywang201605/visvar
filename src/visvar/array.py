"""Antenna arrays and UVW computation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers
import astropy.units as u


@dataclass
class Array:
    """Antenna array described by geocentric (ITRF) positions in metres."""

    names: Sequence[str]
    positions: np.ndarray  # (nant, 3) ITRF metres
    diameter: float = 13.5
    mount: str = "ALT-AZ"
    name: str = "SIMARRAY"
    location: Optional[EarthLocation] = None

    def __post_init__(self):
        self.positions = np.asarray(self.positions, float).reshape(-1, 3)
        self.names = list(self.names)
        if len(self.names) != len(self.positions):
            raise ValueError("names and positions length mismatch")
        if self.location is None:
            c = self.positions.mean(axis=0)
            self.location = EarthLocation.from_geocentric(*c, unit=u.m)

    @property
    def nant(self):
        return len(self.names)

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_enu(cls, lon_deg, lat_deg, height_m, enu, names=None, **kw):
        """Build from local East-North-Up offsets (m) about a reference site."""
        enu = np.asarray(enu, float).reshape(-1, 3)
        ref = EarthLocation.from_geodetic(lon_deg * u.deg, lat_deg * u.deg, height_m * u.m)
        lon, lat = np.radians(lon_deg), np.radians(lat_deg)
        e = np.array([-np.sin(lon), np.cos(lon), 0.0])
        n = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
        up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        r0 = np.array([ref.x.to_value(u.m), ref.y.to_value(u.m), ref.z.to_value(u.m)])
        pos = r0 + enu[:, 0:1] * e + enu[:, 1:2] * n + enu[:, 2:3] * up
        names = names or [f"ANT{i:03d}" for i in range(len(enu))]
        return cls(names, pos, location=ref, **kw)

    @classmethod
    def y_shaped(cls, lon_deg, lat_deg, height_m, n_per_arm=8, arm_length=1000.0, **kw):
        """Simple VLA-like Y layout, handy for tests and demos."""
        enu = [[0.0, 0.0, 0.0]]
        for k in range(3):
            ang = np.radians(90 + 120 * k)
            for i in range(1, n_per_arm + 1):
                r = arm_length * (i / n_per_arm) ** 1.7
                enu.append([r * np.cos(ang), r * np.sin(ang), 0.0])
        return cls.from_enu(lon_deg, lat_deg, height_m, enu, **kw)

    @classmethod
    def from_ms(cls, path):
        from casacore.tables import table
        with table(f"{path}::ANTENNA", ack=False) as t:
            return cls(
                list(t.getcol("NAME")),
                t.getcol("POSITION"),
                diameter=float(np.median(t.getcol("DISH_DIAMETER"))),
                mount=str(t.getcol("MOUNT")[0]),
            )

    @classmethod
    def from_file(cls, path, **kw):
        """Text file with columns ``name x y z`` (ITRF metres)."""
        names, pos = [], []
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            names.append(p[0])
            pos.append([float(v) for v in p[1:4]])
        return cls(names, pos, **kw)


def _itrs_to_gcrs_matrices(mjd_seconds):
    """Rotation matrices GCRS->ITRS (ntime, 3, 3) incl. precession, nutation,
    UT1 and polar motion (IAU 2006/2000A via ERFA)."""
    import erfa
    t = Time(np.atleast_1d(mjd_seconds) / 86400.0, format="mjd", scale="utc")
    # Never hit the network; fall back gracefully outside bundled IERS range.
    with iers.conf.set_temp("auto_download", False), \
            iers.conf.set_temp("iers_degraded_accuracy", "ignore"):
        ut1 = t.ut1
        xp, yp = iers.earth_orientation_table.get().pm_xy(t)
    tt = t.tt
    return erfa.c2t06a(tt.jd1, tt.jd2, ut1.jd1, ut1.jd2,
                       xp.to_value(u.rad), yp.to_value(u.rad))


def antenna_uvw(positions, mjd_seconds, ra0, dec0):
    """UVW (metres) of each antenna for a phase centre (ra0, dec0) (ICRS, rad).

    Returns (ntime, nant, 3).  Baseline ``(p, q)`` UVW (MS convention) is
    ``uvw[q] - uvw[p]``.
    """
    R = _itrs_to_gcrs_matrices(mjd_seconds)                  # (nt, 3, 3)
    pos = np.einsum("tji,aj->tai", R, np.asarray(positions))  # R^T p -> GCRS
    sa, ca, sd, cd = np.sin(ra0), np.cos(ra0), np.sin(dec0), np.cos(dec0)
    eu = np.array([-sa, ca, 0.0])
    ev = np.array([-sd * ca, -sd * sa, cd])
    ew = np.array([cd * ca, cd * sa, sd])
    return np.stack([pos @ eu, pos @ ev, pos @ ew], axis=-1)
