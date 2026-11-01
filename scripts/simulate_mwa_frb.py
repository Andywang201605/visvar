import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

from astropy.utils.iers import conf
conf.auto_max_age = None

import multiprocessing

import visvar

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
    

class MwaSimObs:
    """Simulated MWA observation."""

    CORRS = ["XX", "XY", "YX", "YY"]
    AUTOCORR = False

    def __init__(
            self, metafile, tilefile, maxant=None
        ):
        self.array = self._load_tilefile(tilefile, maxrow=maxant)
        self.metadata = self._load_metafile(metafile)

        self._derive_basic_info()

        
    def _load_tilefile(self, tilefile, maxrow=None):
        """Load MWA tile positions from text file."""
        logger.info(f"Loading MWA tile positions from {tilefile}")
        mwapos = pd.read_csv(tilefile)
        if maxrow is not None:
            logger.info(f"Limiting to first {maxrow} tiles")
            mwapos = mwapos[:maxrow]
        mwaloc = EarthLocation.of_site("mwa")
    
        return visvar.Array.from_enu(
            lon_deg = mwaloc.lon.deg,
            lat_deg = mwaloc.lat.deg,
            height_m = mwaloc.height.value,
            enu = mwapos[["East", "North", "Height"]].to_numpy(),
            names = mwapos["Tilename"].to_numpy(),
        )

    def _load_metafile(self, metafile):
        """Load MWA metadata from FITS header."""
        logger.info(f"Loading MWA metadata from {metafile}")
        return fits.Header.fromstring(
            open(metafile).read(), sep="\n"
        )

    def _derive_baseline(self):
        """Derive baseline information from array."""
        self.nant = self.array.nant
        k = 0 if self.AUTOCORR else 1
        self.a1, self.a2 = np.triu_indices(self.nant, k=k)
        self.nbl = self.a1.size   
        logger.info(f"Number of antennas: {self.nant}; number of baselines: {self.nbl}")

    def _derive_basic_info(self):
        ### load some information from the metadata
        # pointing related
        logger.info(f"pointing centre: RA={self.metadata['RA']}, DEC={self.metadata['DEC']}")
        self.phasecentre = SkyCoord(self.metadata["RA"], self.metadata["DEC"], unit="deg")
        self.ra0 = self.phasecentre.icrs.ra.rad
        self.dec0 = self.phasecentre.icrs.dec.rad
        # channel related
        self.nchan = self.metadata["NCHANS"]
        self.bandwidth = self.metadata["BANDWDTH"] * 1e6
        self.freqcentre = self.metadata["FREQCENT"] * 1e6
        self.df = self.bandwidth / self.nchan
        self.freqs = self.freqcentre - self.bandwidth / 2.0 + np.arange(self.nchan) * self.df
        logger.info(f"frequency range: {self.freqs[0]/1e6:.3f} MHz - {self.freqs[-1]/1e6:.3f} MHz")
        logger.info(f"channel width: {self.df/1e3:.3f} kHz; number of channels: {self.nchan}")
        # baseline related
        self._derive_baseline()

    ####### time related - including uvw calculation
    def calc_uvw(self, times_mjd_s):
        """Calculate uvw coordinates for all baselines at given times."""
        nt = len(times_mjd_s)
        logger.info(f"Calculating uvw coordinates for {nt} time samples")
        logger.info(f"Starting from {times_mjd_s[0]} MJD(UTC) seconds, ending at {times_mjd_s[-1]} MJD(UTC) seconds")

        auvw = visvar.array.antenna_uvw(
            self.array.positions, times_mjd_s, self.ra0, self.dec0
        )  # (nt, nant, 3)
        uvw = (auvw[:, self.a2] - auvw[:, self.a1]).reshape(nt * self.nbl, 3)
        tidx = np.repeat(np.arange(nt), self.nbl)
        return uvw, tidx

    ######## source related
    def generate_burst(self, sim_start_mjd_s, burst_mjd_s, width, peak, dm, ref_freq, ra, dec):
        """Generate a boxcar burst source."""
        t0 = burst_mjd_s - sim_start_mjd_s
        bb = visvar.BoxcarBurst(t0=t0, width=width, peak=peak, dm=dm, ref_freq=ref_freq)
        src = visvar.sky.Source(I=bb, coord=SkyCoord(ra, dec, unit="deg"))
        return src

    ######## visibility prediction
    def predict_visibilities(
            self, src, sim_start_mjd_s, times_mjd_s, nt_sub=10, nf_sub=10, dt=None,
            freqs=None
        ):
        """Predict visibilities for a given source and times."""
        times_mjd_s = np.asarray(times_mjd_s)
        uvw, tidx = self.calc_uvw(times_mjd_s)
        if dt is None:
            dt = times_mjd_s[1] - times_mjd_s[0]  # assuming uniform spacing

        df = self.df if freqs is None else (freqs[1] - freqs[0])
        freqs = self.freqs if freqs is None else freqs
        vis = visvar.predict.predict_visibilities(
            visvar.sky.Sky([src]), self.phasecentre, times_mjd_s - sim_start_mjd_s, dt, tidx,
            uvw, freqs, df, self.CORRS, nt_sub=nt_sub, nf_sub=nf_sub
        )
        return vis

    def predict_visibilities_parallel(
            self, src, sim_start_mjd_s, times_mjd_s, nt_sub=10, nf_sub=10, freqs=None,
            ncpus=None, maxmem=1e10,
        ):
        """Predict visibilities for a given source and times using parallel processing."""
        if ncpus is None:
            ncpus = multiprocessing.cpu_count()

        logger.info(f"Predicting visibilities using {ncpus} CPUs")
        nt = len(times_mjd_s)
        dt = times_mjd_s[1] - times_mjd_s[0]  # assuming uniform spacing
        ### work out chunk size based on memory usage...
        nchan = len(freqs) if freqs is not None else self.nchan
        mem_per_vis = 16  # bytes per complex64
        mem_per_sample = nchan * self.nbl * mem_per_vis
        logger.info(f"Estimated memory usage per sample: {mem_per_sample / 1e6:.2f} MB")
        chunk_size = int(maxmem / mem_per_sample)

        ####### check ncpu constraint...
        if chunk_size > nt // ncpus:
            chunk_size = nt // ncpus
            logger.info(f"Adjusted chunk size to {chunk_size} to respect CPU count")

        logger.info(f"Using chunk size of {chunk_size} time samples per process")
        ###
        chunks = [times_mjd_s[i:i + chunk_size] for i in range(0, nt, chunk_size)]
        with multiprocessing.Pool(processes=ncpus) as pool:
            results = pool.starmap(
                self.predict_visibilities,
                [(src, sim_start_mjd_s, chunk, nt_sub, nf_sub, dt, freqs) for chunk in chunks]
            )
        vis = np.concatenate(results, axis=0)
        return vis



    