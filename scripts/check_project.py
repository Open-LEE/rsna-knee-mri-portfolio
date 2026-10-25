"""Run offline syntax checks and synthetic-data tests from any current directory."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import nbformat

root = Path(__file__).resolve().parents[1]
for path in (root/'src').rglob('*.py'):
    ast.parse(path.read_text(encoding='utf8'), filename=str(path))
for path in (root/'notebooks').glob('*.ipynb'):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            assert not cell.outputs
            ast.parse(cell.source)
env = os.environ.copy()
env['PYTHONPATH'] = str(root/'src')
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
               cwd=root, env=env, check=True)
print('PASS: source / notebooks / synthetic tests. No real GPU training performed.')
