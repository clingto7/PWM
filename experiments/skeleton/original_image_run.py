"""Original image through observation, subject mesh, scene, physics and proxy video."""
import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import observation
import first_image_pipeline
from subject_mesh import verify as verify_mesh
from stage_contracts import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--depth-source', type=Path)
    parser.add_argument('--segmentation-source', type=Path)
    parser.add_argument('--saved-mesh', type=Path)
    parser.add_argument('--mesh-provider', default='hunyuan3d21')
    parser.add_argument('--mesh-weights', type=Path, default=observation.DEFAULT_MODELS/'hunyuan21')
    parser.add_argument('--da3-weights', type=Path, default=observation.DEFAULT_MODELS/'da3nested')
    parser.add_argument('--sam-weights', type=Path, default=observation.DEFAULT_MODELS/'sam21large')
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--resume-observation', type=Path,
                        help='reuse a validated original-image observation and rerun fitting')
    parser.add_argument('--use-weak-prior', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('output directory is not empty')
    root.mkdir(parents=True, exist_ok=True)
    if args.resume_observation:
        source = args.resume_observation.resolve()
        for stage in ('depth', 'segmentation', 'observation', 'downstream_bundle'):
            check(source, stage)
        if observation.digest(args.image) != observation.digest(source/'inputs/input.jpg'):
            raise ValueError('resume observation image differs from requested original')
        shutil.copytree(source, root/'observation')
    else:
        observation.build(SimpleNamespace(image=args.image, output=root/'observation',
            depth_source=args.depth_source, segmentation_source=args.segmentation_source,
            da3_weights=args.da3_weights, sam_weights=args.sam_weights,
            mesh=args.saved_mesh, generate_mesh=True, mesh_provider=args.mesh_provider,
            mesh_weights=args.mesh_weights))
    for stage in ('depth', 'segmentation', 'observation', 'downstream_bundle'):
        check(root/'observation', stage)
    verify_mesh(root/'observation/mesh')
    # Retain the established run CLI and its scene, physics, and renderer contracts.
    first_image_pipeline.main_with_arguments(root/'run', root/'observation', args.seconds,
                                             args.use_weak_prior)
    print(json.dumps({'observation': str(root/'observation'), 'run': str(root/'run')}))


if __name__ == '__main__':
    main()
