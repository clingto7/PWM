# Issue #7 capability probes

Run each probe independently with Python from the first-round A800 environment
(`experiments/first_image/requirements-a800.txt` pins PyBullet 3.2.7 and
gsplat 1.5.3). Choose three distinct output directories outside the checkout:

```sh
python experiments/capabilities/physics.py /path/to/physics-run
python experiments/capabilities/gaussian.py /path/to/gaussian-run
python experiments/capabilities/raster.py /path/to/raster-run
python experiments/capabilities/check_record.py /path/to/physics-run
```

Repeat the last command for each run. Each `run.json` records the invoked
command, versions, elapsed time, measured checks, and output SHA-256. The
checker reads the directory without modifying it. Preserve stdout and stderr
when a probe fails: an import or CUDA failure can occur before a run record is
written. Record that failure, versions, and the alternative before continuing
dependent issues.

The physics record checks the full final second against the flat ground's
expected 0.2 m center height. It reports maximum absolute height error and
maximum ground contact penetration; each has an explicit 0.01 m tolerance.
Both PyBullet probes record the installed package version and the API version.

Capability decisions remain pending the host run. If PyBullet fails the
heightfield or convex hull checks, evaluate MuJoCo's heightfield and convex
mesh support. If gsplat cannot carry four feature channels, use separate
one-hot passes and depth rendering. If TinyRenderer cannot provide headless
depth and segmentation, use an EGL rasterizer with explicit ID material passes.

The probes use synthetic geometry only; no model outputs or assets are needed.
