import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

############## first part - pulse generator ##############

import numpy as np

K_DM = 4.148808e-3  # s GHz^2 cm^3 / pc
C = 299792458.0 # m/s # speed of light

class LightCurve:
    """
    Base class for light curves
    """
    def evaluate(self, t, f):
        """
        Evaluate the light curve at time t and frequency f

        t and f can be vectors, and the output will be a 2D array with shape (len(t), len(f)).
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def sample(self, t, dt, f, df, tsub=10, fsub=10):
        """
        sample the light curve at time t and frequency f, with time resolution dt and frequency resolution df
        take average over tsub and fsub samples to simulate smearing in time and frequency

        Both t and f are vectors, and the output will be a 2D array with shape (len(t), len(f)).
        """
        toff = ((np.arange(tsub) + 0.5) / tsub - 0.5) * dt
        foff = ((np.arange(fsub) + 0.5) / fsub - 0.5) * df
        T = t[:, None, None, None] + toff[None, :, None, None]
        F = f[None, None, :, None] + foff[None, None, None, :]
        vals = self.evaluate(T, F)
        # TODO: need to check axes to be used here
        return vals.mean(axis=(1, 2, 4))

    # TODO: arithmetic operations can be added, but here we just ignore

class BoxCarBurst(LightCurve):
    """
    Boxcar burst light curve
    """
    def __init__(self, t0, width, peak, dm, reffreq=np.inf):
        self.t0, self.width, self.peak = float(t0), float(width), float(peak)
        self.dm, self.reffreq = float(dm), float(reffreq)

    def delay(self, freq):
        """
        Calculate the dispersion delay at a given frequency
        """
        d = K_DM * self.dm * (freq / 1e9) ** -2
        d -= K_DM * self.dm * (self.reffreq / 1e9) ** -2 
        return d

    def evaluate(self, t, f):
        t = t[..., None]
        x = t - self.t0 - self.delay(f)
        return np.where((x >= 0) & (x <= self.width), self.peak, 0.0)

############## second part - lightcurve to visibility mapping ##############
class Source:
    """
    A source with a light curve and a position on the sky
    """
    def __init__(self, lightcurve, ra, dec):
        self.lightcurve = lightcurve
        self.ra, self.dec = float(ra), float(dec)
        self.rarad = np.deg2rad(self.ra)
        self.decrad = np.deg2rad(self.dec)

    def lmn(self, ra0, dec0):
        """
        Calculate the lmn coordinates of the source relative to a phase centre
        """
        ra0rad = np.deg2rad(ra0)
        dec0rad = np.deg2rad(dec0)
        dra = self.rarad - ra0rad
        l = np.cos(self.decrad) * np.sin(dra)
        m = np.sin(self.decrad) * np.cos(dec0rad) - np.cos(self.decrad) * np.sin(dec0rad) * np.cos(dra)
        n = np.sqrt(np.clip(1.0 - l**2 - m**2, 0.0, None))
        return l, m, n

    def dynamic_spectrum(self, t, dt, f, df, corrs, tsub=10, fsub=10):
        """
        Calculate the dynamic spectrum of the source at time t and frequency f
        """
        ntime = t.size; nchan = f.size
        ii = self.lightcurve.sample(t, dt, f, df, tsub=tsub, fsub=fsub)
        vis = np.empty((ntime, nchan, len(corrs)), dtype=np.complex128)
        if corrs == ["XX", "YY"]:
            vis[:, :, 0] = ii
            vis[:, :, 1] = ii
        elif corrs == ["XX", "XY", "YX", "YY"]:
            vis[:, :, 0] = ii
            vis[:, :, 1] = 0.0
            vis[:, :, 2] = 0.0
            vis[:, :, 3] = ii
        elif corrs == ["I"]:
            vis[:, :, 0] = ii
        else:
            raise ValueError(f"Unsupported correlation products: {corrs}")
        return vis
             

def phase_term(uvw, lmn, freqs):
    """
    Calculate the phase term for a given uvw coordinate and lmn position
    uvw: (nrow, 3) array of uvw coordinates in meters
    lmn: (3,) array of lmn coordinates
    freq: (nchan,) array of frequencies in Hz
    returns: (nrow, nchan) array of complex phase terms
    """
    l, m, n = lmn
    path = uvw[:, 0] * l + uvw[:, 1] * m + uvw[:, 2] * (n - 1.0)  # metres
    return np.exp(2j * np.pi * path[:, None] * (np.asarray(freqs)[None, :] / C))
    # return np.exp(2j * np.pi * path * (freq / C))

############## third part - visibility production ##############

def predict_visibilities(
        source, ra0, dec0, times, dt, uvw, freqs, df, corrs, 
        tsub=10, fsub=10, fixuvw=True,
):
    """
    Predict visibilities for a given source and times

    fixuvw: if True, use the same uvw for all times, the number of rows = nbl
    otherwise uvw is different for each time, the number of rows = nbl * ntimes
    """
    ntime = times.size
    if fixuvw: 
        nbl = len(uvw)
        uvw = np.repeat(uvw, ntime, axis=0)
    else:
        assert len(uvw) % ntime == 0, "uvw must be a multiple of ntime"
        nbl = int(len(uvw) / ntime)

    tidx = np.repeat(np.arange(ntime), nbl)
    src = source.dynamic_spectrum(times, dt, freqs, df, corrs, tsub=tsub, fsub=fsub)
    pha = phase_term(uvw, source.lmn(ra0, dec0), freqs)
    return src[tidx] * pha[:, :, None]

def predict_visibilities_tabrow(
        source, ra0, dec0, times, dt, uvw, freqs, df, corrs,
        tsub=10, fsub=10,
):
    """
    predict visibilities based on the casa table rows, i.e., the number of rows of times and the number of rows for uvw are the same...
    """
    src = source.dynamic_spectrum(times, dt, freqs, df, corrs, tsub=tsub, fsub=fsub)
    pha = phase_term(uvw, source.lmn(ra0, dec0), freqs)
    return src * pha[:, :, None]


############## fourth part - visibility injection to the measurement set ##############
from casacore.tables import table

class MeasurementSet:
    """
    A simple class to represent a measurement set
    """
    def __init__(self, mspath):
        self.mspath = mspath
        self._load_info()
        self.tab = table(mspath, readonly=True)

    # load basic information from the measurement set...
    def _load_info(self):
        with table(f"{self.mspath}::SPECTRAL_WINDOW") as spwtab:
            self.freqs = spwtab.getcol("CHAN_FREQ")[0]
            self.df = float(np.mean(np.abs(spwtab.getcol("CHAN_WIDTH")[0])))

        with table(f"{self.mspath}::FIELD") as fieldtab:
            pd = fieldtab.getcol("PHASE_DIR")[0][0]
            self.phase_centre = (np.rad2deg(pd[0]), np.rad2deg(pd[1]))
            self.ra0, self.dec0 = self.phase_centre

        with table(self.mspath) as tab:
            self.times = np.unique(tab.getcol("TIME"))
            self.dt = np.mean(tab.getcol("INTERVAL"))

    # inject source into measurement set
    def inject_source(self, source, tsub=10, fsub=10,):
        self.tab.close()
        tab = table(self.mspath, readonly=False, ack=False)

        nrow = len(tab)
        for i in range(nrow):
            _data = tab.getcol(startrow=i, nrow=1, columnname="DATA")
            _uvw = tab.getcol(startrow=i, nrow=1, columnname="UVW")
            _time = tab.getcol(startrow=i, nrow=1, columnname="TIME")

            vis = predict_visibilities(
                source, self.ra0, self.dec0, _time, self.dt, _uvw,
                self.freqs, self.df, corrs=["XX", "XY", "YX", "YY"],
                tsub=tsub, fsub=fsub,
            )

            data = _data + vis

            tab.putcol(startrow=i, nrow=1, columnname="DATA", value=data.astype(np.complex64))
            tab.flush()
        tab.close()
        self.tab = table(self.mspath, readonly=True)

    def inject_source_batch(self, source, tsub=10, fsub=10, chunk_per_row=1000):
        """
        Inject source into measurement set in batch mode, for speedup
        """
        self.tab.close()
        tab = table(self.mspath, readonly=False, ack=False)

        nrow = len(tab)
        logger.info(f"Injecting source into {nrow} rows of the measurement set...")
        nchunk = nrow // chunk_per_row + (nrow % chunk_per_row > 0)
        logger.info(f"Injecting source in {nchunk} chunks of {chunk_per_row} rows each...")
        for ib in range(nchunk):
            istart = ib * chunk_per_row
            iend = min((ib + 1) * chunk_per_row, nrow)
            nrows = iend - istart

            logger.info(f"Injecting source into rows {istart} to {iend}...")

            _data = tab.getcol(startrow=istart, nrow=nrows, columnname="DATA")
            _uvw = tab.getcol(startrow=istart, nrow=nrows, columnname="UVW")
            _time = tab.getcol(startrow=istart, nrow=nrows, columnname="TIME")

            vis = predict_visibilities_tabrow(
                source, self.ra0, self.dec0, _time, self.dt, _uvw,
                self.freqs, self.df, corrs=["XX", "XY", "YX", "YY"],
                tsub=tsub, fsub=fsub,
            )

            data = _data + vis

            tab.putcol(startrow=istart, nrow=nrows, columnname="DATA", value=data)
            tab.flush()
        tab.close()
        self.tab = table(self.mspath, readonly=True)

if __name__ == "__main__":
    ### test code for injection
    from astropy.coordinates import SkyCoord
    from astropy import units

    mspath = "../data/1218734296_10sec.ms"
    ms = MeasurementSet(mspath)
    coord = SkyCoord("22:08:25.14 -16:13:52.56", unit=(units.hourangle, units.deg))

    source = Source(
        lightcurve = BoxCarBurst(t0=ms.times[0]-1, width=30, peak=1, dm=0, reffreq=1.5e8),
        ra = coord.ra.deg, dec = coord.dec.deg,
    )

    # ms.inject_source(source)
    ms.inject_source_batch(source, chunk_per_row=3000)