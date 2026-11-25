# Issue #8 end-to-end skeleton

Run in the pinned first-image environment, with `ffmpeg` on `PATH`:

```sh
python experiments/skeleton/run.py /path/to/new-run
python experiments/skeleton/check.py /path/to/new-run
```

The run command consumes the existing metric DA3 depth and camera, SAM-derived car mask, and original image. It fits the near-road plane, writes a bounded sector mesh and height map, fills the car footprint with that plane, places a box from the car mask and depth, applies a 3 m/s sliding impulse, simulates in PyBullet, and renders fixed-camera depth and instance ID frames. The PNG depth values are millimetres; ID values are 0 for outside, 1 for ground, 2 for car. Two MP4 files provide contrast-scaled previews. `run.json` lists source and output hashes, parameters, versions, timings, and simulation measurements. The checker only reads files.

Current limitations: the plane is the height map in this skeleton, so local bumps and obstacles are not represented. The box is axis-aligned and only approximates the photographed car. The checker uses provisional invariant tolerances, not the final Spec #6 failure thresholds. A800 validation is required before claiming this run passes the physical and visual acceptance criteria.
