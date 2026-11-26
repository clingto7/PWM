# First-image pipeline skeleton and impulse cases

Run in the pinned first-image environment, with `ffmpeg` on `PATH`:

```sh
python experiments/skeleton/run.py /path/to/new-run
python experiments/skeleton/check.py /path/to/new-run
python experiments/skeleton/conclude.py /path/to/new-run
```

For a visual comparison of all three motions, use `--render-all-cases`.
`--coast-speed` changes only the sliding case; ballistic keeps `--speed`, and
the obstacle case keeps its independently sized launch. For example:

```sh
python experiments/skeleton/run.py /path/to/new-run --coast-speed 7 --render-all-cases --render-gaussians
```

The coast raster videos remain at the run root. Ballistic and obstacle raster
videos and frames live in their respective subdirectories. With Gaussian
rendering enabled, each case also gets Gaussian depth and instance PNG frames.
The Gaussian comparison checker continues to use the coast frames as its
baseline; all additional frames and videos are hashed in the chain record.

The conclusion command writes `first_round_conclusion.json` in the run
directory. Its `content_sha256` hashes canonical JSON without that digest
field, and `source_run_sha256` identifies the run record. Prototype gates in
`gates.py` use the accepted first-image IoU 0.831735 to set a 0.80 minimum
(3.17 percentage points of margin), and the measured 0/52 road false positives
to set a 0.02 maximum (about one road cell). These bounds apply to this scene,
not to general image accuracy. `--gates overrides.json` can set a gate to
`null`, which displays `未设` and never passes. The size reasonableness check
is reported separately from the four failure conditions.

The run writes a coast trajectory at the root and separate
`ballistic/` and `obstacle/` trajectories, each with a replay. `impulses.json`
records the velocity change, impulse magnitude and direction, center of mass
application point, and obstacle target. The checker reports the measured and
expected ballistic flight time and range, coast stopping distance, and obstacle
contact and penetration, plus the shared rest, energy, replay, and camera checks.
The checker reports analytical errors and pre-landing parabola residuals.
After the first complete 5 s / 150-frame A800 run (c4ce582), the six new
limits were backfilled as measured prototype gates:

| Quantity | First measurement | Gate | Basis |
| --- | ---: | ---: | --- |
| Final angular speed | 0 rad/s, all cases | 0.001 rad/s | Allows small numerical residue while requiring near-zero rotation; checked separately from linear speed. |
| Coast distance error | −0.0102186 m | 0.02 m | About 1 cm margin over the measured absolute error. |
| Ballistic flight time error | −0.0119139 s | 0.02 s | About two 1/240 s physics steps beyond the measured error. |
| Ballistic range error | −0.0357416 m | 0.05 m | About 1.4 cm margin over the measured absolute error. |
| Maximum prelanding vertical residual | 0.00800469 m | 0.01 m | About 2 mm margin over the measured residual. |
| Maximum prelanding horizontal residual | 1.67e−14 m | 1e−6 m | Allows floating-point noise while still catching micrometre-scale drift. |

These are bounds for the selected scene and solver, not general accuracy
claims. The existing 0.001 m/s final linear-speed gate is unchanged.
New runs use schema 2 and require all three cases; the checker continues to
accept historical schema 1 runs against their original checks. Linear final
speed and angular final speed are reported separately in m/s and rad/s.
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

Issue #12 comparison is opt-in, preserving CPU-only raster runs:

```sh
python experiments/skeleton/run.py /path/to/new-run --render-gaussians
python experiments/skeleton/check.py /path/to/new-run
```

This option requires CUDA and gsplat 1.5.3 in the pinned A800 environment.
Before launching gsplat, the schema 2 run record declares Gaussian
as a required renderer; the checker fails an incomplete Gaussian run. It writes
`gaussian_depth/`, `gaussian_instance_id/`, and
`gaussian_edge_mixing/` PNGs at the raster frames' 504×280 resolution.
Gaussian depth is gsplat expected optical depth in millimetres; zero denotes
no covered entity. IDs use argmax over one-hot feature channels, with a 0.2
alpha cutoff. The edge-mixing PNG marks pixels where the second ID score is at
least 25% of the strongest and exceeds 0.02. The checker compares each saved
frame against the raster frame, reporting per-ID IoU, depth MAE/max on pixels
covered by both, raster-only crack pixels, mixed-edge pixels, and average
per-frame times. These are measurements, not pass thresholds. Frame timing
includes pose update, rendering, and GPU/CPU readback but excludes PNG writing.
`run.json` records deterministic geometry sampling, per-entity counts, the
depth/feature method, versions, frame times, output hashes, and stage inputs.
gsplat's RGB+ED feature/depth path passed the accepted #7 probe, so no fallback
is active. If this path fails on another gsplat version, record that failure
and use per-entity alpha passes for IDs plus a separate ED depth pass as the
fallback; that alternative is documented but is not implemented in this ticket.

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
