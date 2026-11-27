# First-image pipeline skeleton and impulse cases

Run in the pinned first-image environment, with `ffmpeg` on `PATH`:

```sh
python experiments/skeleton/run.py /path/to/new-run
python experiments/skeleton/check.py /path/to/new-run
python experiments/skeleton/conclude.py /path/to/new-run
```

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

## Portable runs and renderer decision

ADR-0009's accepted addendum makes rasterization the first-round main renderer.
`--render-gaussians` enables optional measured gsplat comparison. The `depth/`
and `instance_id/` PNGs are 16-bit measurements; MP4 palette and contrast
mapping are preview-only. The fitted car-size plausibility check remains
negative in the existing evidence and is reported separately.

A new run copies the four source inputs into `inputs/` and writes `bundle.json`
with their paths, all CLI parameters and the local code revision. To use a
custom bundle, pass `--bundle /path/to/bundle` to `run.py`; its `bundle.json`
must map `image`, `da3`, `sam_car` and `hunyuan_mesh` to files within that
bundle. To reproduce on a machine with that exact code revision and the pinned
first-image environment (including ffmpeg), copy only the run directory and run:

```sh
python experiments/skeleton/reproduce.py /path/to/copied-run /path/to/new-run
python experiments/skeleton/check.py /path/to/new-run
```

The original input paths are unnecessary. The checker still accepts historical
runs whose input entries contain absolute paths. New runs contain separate
`raster_replay/depth/` and `raster_replay/instance_id/` PNGs rendered from the
saved trajectory with the same fixed camera; the checker compares PNG bytes
frame by frame. Gaussian replay, when enabled, records measured differences
rather than claiming bitwise determinism.

## Module interfaces

`scene.reconstruct(inputs, args, output, start)` consumes the four bundled
inputs and run settings, writes the explicit `SceneArtifacts` set, and returns a `Reconstruction` containing the saved artifact set and
construction timings. `artifacts.SceneArtifacts.load()` rebuilds collision and visual geometry from
portable files without the original model inputs. `SceneArtifacts` validates
the saved set: `scene.json`, `heightmap.npz`,
`obstacles.npz`, `fitted_car.obj`, and `car_collision_hull.obj`.
`physics.PyBulletSolver.create_world(scene_artifacts.load(), settings, output)` creates
the static collision world and vehicle; `solve(body, obstacle_body_ids,
delta_velocity, seconds)` returns states, maximum penetration, and obstacle
contact counts. A replacement solver implements these two operations. `raster.RasterRenderer.render(trajectory,
depth_dir, instance_id_dir, preview_writers=None)` writes 16-bit proxy frames
from a fixed camera. `run.py` selects `RasterRenderer` unconditionally as the
mainline adapter and invokes gsplat only with `--render-gaussians`.

To re-render only the saved scene and trajectory (without reconstruction or
simulation), use:

```sh
python experiments/skeleton/raster_replay.py /path/to/run /path/to/empty-replay-output
```

`run.py` uses this same saved-artifact path for its separate raster replay.
Bundle source hashes are checked before reproduction, and reproduction output
must be outside the source bundle.

New heightmaps retain the historical float32 layers and add float64 `ground_z_exact_m` and `surface_z_exact_m` for byte-exact raster replay. `SceneArtifacts.load()` falls back to float32 layers for historical runs.

## Ticket #15 stage contracts

`run.py` is the first-image entry point. It retains the existing `--bundle` input
and output layout. A completed run writes versioned `stages/*.json` contracts
for observation, subject mesh, scene, actions, physics and raster rendering.
Each contract records the implementation, version, parameters, wall-clock
seconds and SHA-256 hashes of its inputs and outputs. The scene contract also
declares that the saved DA3 depth is metric and includes camera intrinsics.
Validate a saved boundary without running any model or solver:

```sh
python experiments/skeleton/stage_contracts.py check /path/to/run scene
```

To reuse a validated scene with the same input bundle, run:

```sh
python experiments/skeleton/run.py /path/to/new-run --bundle /path/to/old-run --resume-scene /path/to/old-run
```

The command verifies the scene contract and source hashes before copying scene
artifacts. It then runs the saved-scene physics and raster path. A failed run
writes `stage_status.json` with the failing stage and reason. The read-only
`check.py` verifies new contracts when present and keeps legacy runs compatible.
The first-image stage migration uses saved DA3/SAM/Hunyuan outputs; ticket 16
adds original-image DA3 and SAM inference below.

Scene resume rejects any changed scene geometry or obstacle setting recorded in
the saved scene contract (for example `--cell` or `--radius`). Physics now writes
`contacts.json` with a versioned per-case contact and penetration record; the
physics contract hashes it, and the checker compares it with the run record.
`physics.solve_saved_scene` calls `physics.PyBulletSolver` for world creation
and state transitions; `first_image_stages.py` calls the raster renderer, while
`first_image_pipeline.py` coordinates stage calls and writes provenance.

# Ticket 16: original-image observation

`observation.py` accepts an original screenshot and emits checked depth,
segmentation, and observation artifacts without requiring a mesh. With local
DA3 Nested and SAM 2.1 Large weights at
`experiments/first_image/models/{da3nested,sam21large}`:

```sh
python experiments/skeleton/observation.py IMAGE.jpg OBSERVATION_DIR
python experiments/skeleton/observation.py check OBSERVATION_DIR observation
```

For the current-image downstream proxy, add a saved mesh to create the bundle
consumed by `first_image_pipeline.py`:

```sh
python experiments/skeleton/observation.py IMAGE.jpg PROXY_OBSERVATION_DIR --mesh SAVED_MESH.glb
python experiments/skeleton/first_image_pipeline.py RUN_DIR --bundle PROXY_OBSERVATION_DIR
```

`--mesh` supplies a saved mesh only for the current-image downstream proxy;
automatic subject mesh generation belongs to ticket 17. For local development
without model inference, pass `--depth-source` and `--segmentation-source` to
directories containing saved `run.json` and model output files. Each slot can
be changed independently. The observation saves all SAM candidates and scores
in `selection.json`, the selected mask, a depth capability declaration, model
identities, weights hashes, elapsed time, and hash-checked stage records. A
relative-depth or missing-camera result fails before creating an accepted
observation.

## Cross-image automatic chain and evaluation protocol (#22)

Run with one original screenshot and a JSON config:

```sh
python experiments/skeleton/automatic_chain.py run IMAGE.jpg config.json RUN_DIR
python experiments/skeleton/automatic_chain.py check RUN_DIR
python experiments/skeleton/automatic_chain.py replay RUN_DIR NEW_DIR --from-stage scene
python experiments/skeleton/automatic_chain.py replay RUN_DIR RENDER_DIR --from-stage render --renderer raster
```

Config schema 1 accepts `renderer` (`raster` or `gaussian`), `seconds`,
`mesh_provider` (`hunyuan3d21`), `use_weak_prior`, and optional saved model
sources or local weight paths (`depth_source`, `segmentation_source`,
`saved_mesh`, `da3_weights`, `sam_weights`, `mesh_weights`). Without saved
sources, installed local weights are required. No download is attempted.
`chain.json` records the requested configuration, input hashes, selected
stage contracts and implementations, output hashes, wall-clock time and the
stage of any failure. The checker reads only. Observation replay reuses checked
observation; scene replay also reuses the checked scene; physics/render replay
consumes the saved scene and trajectory through the renderer adapter. Renderer
comparisons therefore use identical upstream hashes. Replay output must be
separate and empty. The proxy adapter emits source and nearby-view depth and
entity-ID PNGs plus MP4s; nearby views are geometry diagnostics.

The cross-game pilot is pending source screenshots. Prepare six distinct game
IDs: three development and three holdout, with at least five different eligible
third-person scenes per game and varied dominant subject shapes. Eligibility
requires a visible main support and one dominant controlled subject; decide
eligibility before evaluating a run. Store image paths and scoring-only 2D
annotations in a schema-1 dataset JSON accepted by `evaluation.py`. The
annotation may cover subject mask/bbox, visible support, entities and source-view
render comparisons; it must never enter the runner config. Mark every eligible
scene, including failed runs. `evaluation.py` rejects undersized splits and
counts each missing or failed run as a failure.

Run development first. Before viewing holdout images, freeze the code revision,
config hash, per-image failure conditions, five numeric thresholds (detection
IoU minimum, geometry depth MAE maximum in metres, contact penetration maximum
in metres, render IoU minimum, total seconds maximum), and aggregate success
minimum in a versioned freeze JSON. Commit that freeze JSON before opening
holdout. Use `evaluation.py DATASET --freeze FREEZE --split holdout` only after
that commit. Report per-game and per-scene failures, all five metrics, and the
eligible denominator. The current first-image gates are not cross-game
thresholds; no values are frozen yet and no cross-game trial is claimed.
