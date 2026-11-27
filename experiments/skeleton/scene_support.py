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


def _largest_component(mask, anchor):
    """Return the largest connected inlier patch, preferring subject contact."""
    h,w=mask.shape
    seen=np.zeros_like(mask)
    best=None
    for sy,sx in np.argwhere(mask):
        if seen[sy,sx]: continue
        stack=[(int(sy),int(sx))]; seen[sy,sx]=True; cells=[]; contact=0
        while stack:
            y,x=stack.pop(); cells.append((y,x)); contact+=int(anchor[y,x])
            for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True; stack.append((ny,nx))
        score=(contact>0, len(cells), contact)
        if best is None or score>best[0]: best=(score,cells)
    result=np.zeros_like(mask)
    if best:
        ys,xs=zip(*best[1]); result[ys,xs]=True
    return result,best[0] if best else (False,0,0)


def select_support_plane(xx, yy, zz, valid, subject, tolerance):
    """Deterministic robust plane from image-wide depth and connected support."""
    valid=np.asarray(valid,bool)&np.isfinite(xx)&np.isfinite(yy)&np.isfinite(zz)
    if np.count_nonzero(valid)<100: raise ValueError('insufficient finite support candidates')
    subject=np.asarray(subject,bool)
    # Subject-adjacent evidence helps distinguish its support from distant scenery.
    anchor=np.zeros_like(subject)
    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
        source=subject[max(0,-dy):subject.shape[0]-max(0,dy),max(0,-dx):subject.shape[1]-max(0,dx)]
        anchor[max(0,dy):subject.shape[0]-max(0,-dy),max(0,dx):subject.shape[1]-max(0,-dx)] |= source
    anchor &= valid
    sample=np.flatnonzero(valid.ravel())[::max(1,np.count_nonzero(valid)//4000)]
    points=np.stack((xx.ravel()[sample],yy.ravel()[sample],zz.ravel()[sample]),axis=1)
    rng=np.random.default_rng(18)
    best=None
    for _ in range(160):
        ids=rng.choice(len(points),3,replace=False)
        design=np.column_stack((points[ids,:2],np.ones(3)))
        if abs(np.linalg.det(design))<1e-4: continue
        coeff=np.linalg.solve(design,points[ids,2])
        if not np.all(np.isfinite(coeff)) or np.linalg.norm(coeff[:2])>3: continue
        inliers=valid & (np.abs(zz-(coeff[0]*xx+coeff[1]*yy+coeff[2]))<=tolerance)
        patch,score=_largest_component(inliers,anchor)
        # Contact first, then connected visible area. A broad disconnected
        # background cannot win by accumulating unrelated pixels.
        if score[1]<100: continue
        rank=(score[0],score[1],score[2])
        if best is None or rank>best[0]: best=(rank,patch)
    if best is None: raise ValueError('no connected image-wide support plane')
    patch=best[1]
    design=np.column_stack((xx[patch],yy[patch],np.ones(int(patch.sum()))))
    coeff=np.linalg.lstsq(design,zz[patch],rcond=None)[0]
    inliers=valid & (np.abs(zz-(coeff[0]*xx+coeff[1]*yy+coeff[2]))<=tolerance)
    patch,_=_largest_component(inliers,anchor)
    if patch.sum()<100: raise ValueError('insufficient connected support inliers')
    design=np.column_stack((xx[patch],yy[patch],np.ones(int(patch.sum()))))
    coeff=np.linalg.lstsq(design,zz[patch],rcond=None)[0]
    rms=float(np.sqrt(np.mean((design@coeff-zz[patch])**2)))
    return coeff,patch,rms
