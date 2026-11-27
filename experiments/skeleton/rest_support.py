"""Read-only oriented hull clearance against saved support triangles."""
import math
import numpy as np


def load_obj_vertices(path):
    vertices=[]
    for line in path.read_text().splitlines():
        if line.startswith('v '):
            fields=line.split()
            vertices.append([float(value) for value in fields[1:4]])
    if not vertices: raise ValueError('collision hull has no vertices')
    return np.asarray(vertices,dtype=float)


def hull_support_clearance(row, vertices, heights, valid, cell):
    """Minimum vertical hull clearance at vertices above valid ground triangles."""
    qx,qy,qz,qw=row[3:7]
    rotation=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                       [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                       [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
    points=np.asarray(vertices)@rotation.T+np.asarray(row[:3])
    n=heights.shape[0]; origin=-(n-1)/2*cell
    clearances=[]
    for x,y,z in points:
        gx=(x-origin)/cell;gy=y/cell
        i=int(math.floor(gx));j=int(math.floor(gy))
        if not (0<=i<n-1 and 0<=j<n-1 and valid[j:j+2,i:i+2].all()):continue
        u=gx-i;v=gy-j
        # Saved mesh triangles: (0,0)-(1,0)-(0,1), (1,0)-(1,1)-(0,1).
        if u+v<=1:
            ground=(1-u-v)*heights[j,i]+u*heights[j,i+1]+v*heights[j+1,i]
        else:
            ground=(1-v)*heights[j,i+1]+(u+v-1)*heights[j+1,i+1]+(1-u)*heights[j+1,i]
        clearances.append(float(z-ground))
    return min(clearances) if clearances else None
