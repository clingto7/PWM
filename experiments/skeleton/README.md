# First-image pipeline skeleton and impulse cases

Run in the pinned first-image environment, with `ffmpeg` on `PATH`:

```sh
python experiments/skeleton/run.py /path/to/new-run
python experiments/skeleton/check.py /path/to/new-run
```

The run writes a coast trajectory at the root and separate
`ballistic/` and `obstacle/` trajectories, each with a replay. `impulses.json`
records the velocity change, impulse magnitude and direction, center of mass
application point, and obstacle target. The checker reports the measured and
expected ballistic flight time and range, coast stopping distance, and obstacle
contact and penetration, plus the shared rest, energy, replay, and camera checks.
The checker uses provisional 0.05 s flight and 0.5 m range/distance gates until
the first A800 measurements can set evidence-based limits.

Obstacle options: `--obstacle-height 0.35` (metres above the fitted plane),
`--obstacle-min-pixels 3` (DA3 votes per grid cell), and
`--obstacle-min-cells 2` (four-connected component size). The
`--obstacles-render-only` switch keeps obstacle meshes and instance IDs in the
proxy render while excluding their collision shapes. `heightmap.npz` stores
separate ground and observed surface height layers, surface vote counts, and
provenance (1 observed DA3, 2 fitted plane, 3 car fill). `obstacles.npz` stores
the component ID grid, candidate cells, road denominator, and car footprint.
`scene.json` and `run.json` record the settings, each component's cell count,
the road false positive ratio, and actual PyBullet collision shape counts. The
checker recomputes those measurements from the saved grid and checks every
obstacle ID across all rendered frames, reporting any missing IDs.

The run command consumes the existing metric DA3 depth and camera, SAM-derived car mask, and original image. It fits the near-road plane, forms height-map cells from near metric depth back-projections and far ray/plane intersections, records per-cell provenance, fills the car footprint with that plane, places a box from the car mask and depth, applies a 3 m/s sliding impulse, simulates in PyBullet, and renders fixed-camera depth and instance ID frames. The PNG depth values are millimetres for visible geometry; zero marks no entity or invalid depth. ID values are 0 for outside, 1 for ground, 2 for car. Frame 0 is stationary before the impulse. `initial_projection.json` stores pixel-run evidence from that rendered frame and the resized SAM mask; the checker recomputes bbox/IoU and checks improvement against the published 6075606 baseline IoU of 0.3749. Two MP4 files provide contrast-scaled previews. `run.json` lists source and output hashes, per-stage input/output hashes, parameters, versions, timings, and simulation measurements. The checker only reads files.

Current limitations: only trusted near-road pixels contribute local ground height variation; remaining cells use the fitted plane. Roadside obstacle surfaces use raw DA3 depth within the trusted radius, so far-depth compression can affect detection. The box is axis-aligned and only approximates the photographed car. Checker thresholds were backfilled after A800 run 6075606: 0.01 m penetration, 0.01 m rest-height error, 0.001 m/s final speed, 0.001 J/kg maximum step energy gain, and 0.01 m ground-fit RMS. Each is a simple bound above that first observation, not a statistical accuracy claim. Replay now starts in a fresh physics world. A800 validation is required before claiming this run passes the physical and visual acceptance criteria.
