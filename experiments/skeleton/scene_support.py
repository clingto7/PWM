"""Solver-neutral support evidence and local occlusion completion."""
import numpy as np


def classify_support(observed, occluded, possible, measured_z, plane_z):
    """Codes: 0 unknown, 1 measured, 2 locally interpolated behind subject."""
    observed=np.asarray(observed,dtype=bool) & possible
    occluded=np.asarray(occluded,dtype=bool) & possible & ~observed
    z=np.asarray(measured_z,dtype=float).copy()
    codes=np.zeros(observed.shape,dtype=np.uint8)
    codes[observed]=1
    # Complete only small holes bounded by nearby measured support.
    for y,x in np.argwhere(occluded):
        near=observed[max(0,y-2):y+3,max(0,x-2):x+3]
        values=z[max(0,y-2):y+3,max(0,x-2):x+3][near]
        if values.size >= 2:
            z[y,x]=float(np.median(values))
            codes[y,x]=2
    return codes,codes>0,z
