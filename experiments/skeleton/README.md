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
The checker reports analytical errors and pre-landing parabola residuals with
the acceptance gate marked `未设` until the first A800 measurements establish
evidence-based limits. An unset gate does not count as a pass.
New runs use schema 2 and require all three cases; the checker continues to
accept historical schema 1 runs against their original checks. Linear final
speed keeps its measured m/s gate. Angular final speed is reported separately
in rad/s with its gate unset until the first A800 measurement.
PyBullet combines the ground and car lateral friction multiplicatively. The
ground uses the nominal coefficient and the car uses 1.0, so the effective
coast coefficient matches the recorded nominal value. The obstacle impulse
is sized from estimated distance to first contact plus a small arrival speed,
avoiding the large overspeed seen in the first A800 measurement.
The default obstacle minimum launch speed is 6 m/s, independently of the
3 m/s coast and ballistic speed; this was the lowest measured speed that
contacted the selected obstacle in the second A800 diagnostic. PyBullet's
normal rigid-body sleeping is explicitly enabled and recorded for the car so
terminal contact jitter can settle without editing trajectory velocities.
The existing 0.001 m/s final linear-speed gate remains unchanged.

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

The run command consumes the existing metric DA3 depth and camera, SAM-derived car mask, and original image. It fits the near-road plane, forms height-map cells from near metric depth back-projections and far ray/plane intersections, records per-cell provenance, fills the car footprint with that plane, fits the Hunyuan vehicle mesh from the car mask and depth, applies a 3 m/s sliding impulse, simulates in PyBullet, and renders fixed-camera depth and instance ID frames. The PNG depth values are millimetres for visible geometry; zero marks no entity or invalid depth. ID values are 0 for outside, 1 for ground, 2 for car, and 3 onward for obstacle components. Frame 0 is stationary before the impulse. `initial_projection.json` stores pixel-run evidence from that rendered frame and the resized SAM mask; the checker recomputes bbox/IoU and checks improvement against the published 6075606 baseline IoU of 0.3749. Two MP4 files provide contrast-scaled previews. `run.json` lists source and output hashes, per-stage input/output hashes, parameters, versions, timings, and simulation measurements. The checker only reads files.

Issue #9 replaces the surrogate box with the local Hunyuan3D GLB. The fitter estimates uniform scale, ground-plane yaw, and translation from the SAM silhouette, DA3 car depth, and fitted ground contact. The source GLB and fitted OBJ are hashed in `run.json`; `scene.json` records the fitted pose, size and a post-fit comparison against BMW's published MINI Cooper S Convertible dimensions. The collision OBJ is the fitted mesh's convex hull. PyBullet uses the fitted mesh for visual rendering and the hull for collision at the same body pose. The checker compares the first rendered ID PNG to the recorded silhouette runs. The fit optimizes a projected convex silhouette, so the rendered mesh's measured IoU is the final image evidence.

Current limitations: only trusted near-road pixels contribute local ground height variation; remaining cells use the fitted plane. Roadside obstacle surfaces use raw DA3 depth within the trusted radius, so far-depth compression can affect detection. Checker thresholds were backfilled after A800 run 6075606: 0.01 m penetration, 0.01 m rest-height error, 0.001 m/s final speed, 0.001 J/kg maximum step energy gain, and 0.01 m ground-fit RMS. Each is a simple bound above that first observation, not a statistical accuracy claim. Replay now starts in a fresh physics world. A800 validation is required before claiming this run passes the physical and visual acceptance criteria.

Issue #9 A800 evidence on published `98a27b7`: at 3 s the checker failed only the inherited final-speed gate (0.00228612 m/s versus 0.001 m/s); at 5 s it passed all checks, with 150 frames and final speed 0.000359783 m/s. The default duration is therefore 5 s, with the original gate retained. Initial car mask IoU was 0.831735; ground RMS 0.007185 m, penetration 0.004373 m, resting-height error 0.001524 m, and energy gain 0.00007393 J/kg. Replay was identical, obstacle IDs 3–6 were visible, and the road false-positive count was 0/52.

The size plausibility check remains **false**: the fitted mesh is [1.149, 2.330, 0.898] m, versus BMW's [1.727, 3.876, 1.415] m, or 0.665/0.601/0.635 of the reference dimensions. DA3's median car-body depth is 3.037 m and the mask-bottom ground ray meets the plane at 3.106 m. These depth and silhouette constraints yield the smaller scale; the data do not establish whether depth scale, camera/ground geometry, or generated shape proportions dominate the discrepancy. The fit continues to use image evidence only, and the failed reasonableness result stays in the chain record for review.
