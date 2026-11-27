"""Hash Gaussian frame outputs in their owning directory namespace."""
import hashlib


def hash_outputs(root, base, directories, external=False):
    anchor = base if external else root
    return {str(path.relative_to(anchor)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in directories for path in sorted(directory.glob('*.png'))}
