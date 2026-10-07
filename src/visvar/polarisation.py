"""Stokes <-> correlation products for ideal feeds."""
import numpy as np

CORR_CODES = {"I": 1, "Q": 2, "U": 3, "V": 4,
              "RR": 5, "RL": 6, "LR": 7, "LL": 8,
              "XX": 9, "XY": 10, "YX": 11, "YY": 12}

_FEED = {"XX": "X", "XY": "X", "YX": "Y", "YY": "Y",
         "RR": "R", "RL": "R", "LR": "L", "LL": "L"}


def corr_to_feeds(corrs):
    """Feed types (e.g. ['X','Y']) implied by a correlation list."""
    feeds = []
    for c in corrs:
        f = _FEED.get(c)
        if f and f not in feeds:
            feeds.append(f)
    return feeds or ["X", "Y"]


def stokes_to_corr(corr, I, Q, U, V):
    """Visibility-plane coherency for correlation ``corr``."""
    c = corr.upper()
    if c == "XX": return I + Q
    if c == "YY": return I - Q
    if c == "XY": return U + 1j * V
    if c == "YX": return U - 1j * V
    if c == "RR": return I + V
    if c == "LL": return I - V
    if c == "RL": return Q + 1j * U
    if c == "LR": return Q - 1j * U
    if c in ("I", "Q", "U", "V"):
        return {"I": I, "Q": Q, "U": U, "V": V}[c]
    raise ValueError(f"unsupported correlation {corr!r}")


# MS CORR_PRODUCT pairs (receptor indices) for the cross-hand layout
def corr_product(corrs):
    idx = {"X": 0, "R": 0, "Y": 1, "L": 1}
    out = []
    for c in corrs:
        if c in _FEED:
            out.append([idx[c[0]], idx[c[1]]])
        else:
            out.append([0, 0])
    return np.array(out, dtype=np.int32)
