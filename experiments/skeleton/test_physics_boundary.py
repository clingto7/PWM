"""Physics adapter ownership contract for the saved-scene pipeline."""
import ast
import unittest
from pathlib import Path


class PhysicsBoundaryTest(unittest.TestCase):
    def test_only_adapter_imports_bullet_or_builds_world(self):
        root = Path(__file__).parent
        for name in ('scene.py', 'first_image_stages.py', 'raster_replay.py',
                     'first_image_pipeline.py', 'raster.py'):
            with self.subTest(name=name):
                tree = ast.parse((root/name).read_text())
                imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
                self.assertFalse(any('pybullet' in ast.unparse(node).lower() for node in imports))
                self.assertFalse(any(isinstance(node, ast.Name) and node.id == 'PyBulletSolver' for node in ast.walk(tree)))
                self.assertNotIn('create_world(', (root/name).read_text())


if __name__ == '__main__':
    unittest.main()
