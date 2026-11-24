"""Render four neutral-material views of a generated GLB in headless Blender.

blender -b --python render_mesh.py -- car_shape.glb output_directory
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


args = sys.argv[sys.argv.index("--") + 1:]
mesh_file, output_dir = Path(args[0]), Path(args[1])
output_dir.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(mesh_file))
objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if not objects:
    raise RuntimeError("GLB contains no mesh")

points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
minimum = Vector([min(point[axis] for point in points) for axis in range(3)])
maximum = Vector([max(point[axis] for point in points) for axis in range(3)])
center = (minimum + maximum) / 2
radius = max(maximum - minimum) * 2.2

material = bpy.data.materials.new("neutral_mesh")
material.diffuse_color = (0.7, 0.75, 0.8, 1)
material.use_nodes = True
nodes = material.node_tree.nodes
nodes.clear()
shader = nodes.new("ShaderNodeBsdfPrincipled")
shader.inputs["Base Color"].default_value = (0.7, 0.75, 0.8, 1)
surface = nodes.new("ShaderNodeOutputMaterial")
material.node_tree.links.new(shader.outputs["BSDF"], surface.inputs["Surface"])
for obj in objects:
    obj.data.materials.clear()
    obj.data.materials.append(material)

camera_data = bpy.data.cameras.new("camera")
camera = bpy.data.objects.new("camera", camera_data)
bpy.context.collection.objects.link(camera)
bpy.context.scene.camera = camera
camera_data.type = "ORTHO"
camera_data.ortho_scale = max(maximum - minimum) * 1.7

for position in [(3, 4, 6), (-4, -3, 5)]:
    data = bpy.data.lights.new("area_light", "AREA")
    light = bpy.data.objects.new("area_light", data)
    bpy.context.collection.objects.link(light)
    light.location = center + Vector(position) * max(maximum - minimum)
    data.energy = 450
    data.shape = "DISK"
    data.size = 5

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.render.resolution_x = 600
scene.render.resolution_y = 600
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.world.color = (0.7, 0.7, 0.7)
scene.render.image_settings.file_format = "PNG"

for degrees in (0, 90, 180, 270):
    angle = math.radians(degrees)
    camera.location = center + Vector((math.cos(angle) * radius, math.sin(angle) * radius, radius * 0.38))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(output_dir / f"view_{degrees:03d}.png")
    bpy.ops.render.render(write_still=True)
