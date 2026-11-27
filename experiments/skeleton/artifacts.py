"""Saved scene reconstruction interface consumed by solver and renderers.

A complete scene contains the camera and fitted geometry in scene.json, two
height layers, obstacle IDs, and fitted visual/collision car meshes. Paths are
relative to a single run directory so the artifact set can be moved intact.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneArtifacts:
    root: Path

    NAMES = ('scene.json', 'heightmap.npz', 'obstacles.npz',
             'fitted_car.obj', 'car_collision_hull.obj')

    def files(self):
        paths = {name: self.root / name for name in self.NAMES}
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise ValueError(f'incomplete reconstructed scene: {missing}')
        return paths

    def load(self):
        """Load the saved collision and visual geometry without source inputs."""
        import json
        import math
        from types import SimpleNamespace
        import numpy as np
        import trimesh
        paths=self.files()
        scene=json.loads(paths['scene.json'].read_text())
        with np.load(paths['heightmap.npz']) as heightmap:
            heights=heightmap['ground_z_exact_m'] if 'ground_z_exact_m' in heightmap else heightmap['ground_z_m']
            valid=heightmap['traversable'].astype(bool) if 'traversable' in heightmap else heightmap['valid'].astype(bool)
            surface=heightmap['surface_z_exact_m'] if 'surface_z_exact_m' in heightmap else heightmap['surface_z_m']
        with np.load(paths['obstacles.npz']) as obstacle_data:
            component_ids=obstacle_data['instance_id']
        cell=float(scene['ground']['cell_m']); n=heights.shape[0]
        origin=-(n-1)/2*cell
        vertices=[[origin+i*cell,j*cell,float(heights[j,i])]
                  for j in range(n) for i in range(n)]
        indices=[]
        for j in range(n-1):
            for i in range(n-1):
                if valid[j:j+2,i:i+2].all():
                    q=j*n+i
                    indices.extend((q,q+1,q+n,q+1,q+n+1,q+n))
        components=scene['obstacles']['components']
        flat_z=float(scene['ground']['flat_world_z_m'])
        obstacle_meshes=[]
        for component in components:
            mesh_vertices=[];mesh_indices=[]
            for cy,cx in np.argwhere(component_ids==component['instance_id']):
                x=origin+cx*cell;y=cy*cell
                bottom=flat_z;top=float(surface[cy,cx]);half=cell/2
                q=len(mesh_vertices)
                mesh_vertices.extend([[x+dx*half,y+dy*half,z] for z in (bottom,top) for dy in (-1,1) for dx in (-1,1)])
                mesh_indices.extend(q+t for t in (0,2,1,1,2,3,4,5,6,5,7,6,0,1,4,1,5,4,2,6,3,3,6,7,0,4,2,2,4,6,1,3,5,3,7,5))
            obstacle_meshes.append((mesh_vertices,mesh_indices))
        car=scene['car']
        hull=trimesh.load(paths['car_collision_hull.obj'],force='mesh',process=False)
        return SimpleNamespace(vertices=vertices,indices=indices,
            components=components,obstacle_meshes=obstacle_meshes,
            traversable=valid, cell_m=cell, origin_x=origin,
            half_angle=math.radians(scene['sector']['angle_deg']/2),
            flat_z=flat_z,hull=hull,mass=car['mass_kg'],
            car_pos=car['initial_position_m'],
            car_orientation=car['initial_orientation_xyzw'],scene=scene)
