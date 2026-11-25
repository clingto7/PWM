# Source this file before running first-round inference on the A800 host.
# The two source snapshots are official upstream revisions, copied into the
# assigned workspace without modifying them.
PWM_ASSIGNED_ROOT=/KAIROS_vepfs-2/KAIROS_vepfs/liwenhao/agent/666777
PWM_PYTHON="$PWM_ASSIGNED_ROOT/envs/pwm-first-round/bin/python"
PWM_UV="$PWM_ASSIGNED_ROOT/tools/uv/bin/uv"
PWM_DA3_SOURCE="$PWM_ASSIGNED_ROOT/sources/Depth-Anything-3-3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src"
PWM_HY3D_SOURCE="$PWM_ASSIGNED_ROOT/sources/Hunyuan3D-2.1-82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape"
export PWM_ASSIGNED_ROOT PWM_PYTHON PWM_UV PWM_DA3_SOURCE PWM_HY3D_SOURCE
export PYTHONPATH="$PWM_DA3_SOURCE:$PWM_HY3D_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWM_ASSIGNED_ROOT/cache/hf"
export UV_CACHE_DIR="$PWM_ASSIGNED_ROOT/cache/uv"
export TORCH_EXTENSIONS_DIR="$PWM_ASSIGNED_ROOT/cache/torch_extensions"
export XDG_CACHE_HOME="$PWM_ASSIGNED_ROOT/cache/xdg"
export TMPDIR="$PWM_ASSIGNED_ROOT/cache/tmp"
