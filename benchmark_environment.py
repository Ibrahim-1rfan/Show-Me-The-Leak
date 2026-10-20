"""Isolated model environments: never replace packages in the notebook kernel."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON_VERSION = '3.11'
PROFILES = {
    'vlm': {'torch': '2.8.0', 'torchvision': '0.23.0', 'cuda_index': 'cu128',
            'packages': ['transformers==4.57.1', 'accelerate>=1.11,<2', 'bitsandbytes>=0.48,<1']},
    'chandra': {'torch': '2.8.0', 'torchvision': '0.23.0', 'cuda_index': 'cu128',
                'packages': ['transformers==4.57.1', 'chandra-ocr==0.1.7', 'accelerate>=1.11,<2', 'bitsandbytes>=0.48,<1']},
    'dots': {'torch': '2.7.0', 'torchvision': '0.22.0', 'cuda_index': 'cu128',
             'packages': ['transformers==4.56.1', 'qwen-vl-utils>=0.0.14', 'accelerate>=1,<2',
                          'huggingface-hub<1', 'packaging', 'ninja', 'wheel', 'psutil']},
    'florence': {'torch': '2.6.0', 'torchvision': '0.21.0', 'cuda_index': 'cu124',
                 'packages': ['transformers==4.41.2', 'tokenizers==0.19.1', 'numpy<2',
                              'accelerate>=0.30,<2', 'einops>=0.8,<1', 'timm==1.0.7', 'huggingface-hub<1']},
}
COMMON = ['Pillow>=10', 'pandas>=2', 'matplotlib>=3.8', 'ipython', 'setuptools', 'packaging']


def environment_python(env_dir):
    return Path(env_dir) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def install_commands(profile, env_root, uv_command=None):
    """Pure command plan, also used by tests. GPU stacks stay in separate venvs."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    spec = PROFILES[profile]
    env_dir = Path(env_root) / f'{profile}-py311'
    python = environment_python(env_dir)
    uv = list(uv_command or [sys.executable, '-m', 'uv'])
    pip = [str(python), '-m', 'pip', 'install', '--disable-pip-version-check']
    constraints = env_dir / 'benchmark-constraints.txt'
    commands = [
        [*uv, 'python', 'install', PYTHON_VERSION],
        [*uv, 'venv', '--python', PYTHON_VERSION, '--seed', '--allow-existing', str(env_dir)],
        [*pip, f"torch=={spec['torch']}", f"torchvision=={spec['torchvision']}",
         '--index-url', f"https://download.pytorch.org/whl/{spec['cuda_index']}"],
        [*pip, '-c', str(constraints), '--only-binary=tokenizers', *COMMON, *spec['packages']],
    ]
    if profile == 'dots':
        commands.append([*pip, '-c', str(constraints), 'flash-attn==2.8.0.post2', '--no-build-isolation'])
    return python, constraints, commands


def probe_environment(python, profile, require_gpu=True):
    """Use the worker interpreter; package installation alone is not success."""
    spec = PROFILES[profile]
    expected_tf = next(x.split('==')[1] for x in spec['packages'] if x.startswith('transformers=='))
    script = f'''
import json, sys, torch, torchvision, transformers, tokenizers
from transformers import AutoProcessor
assert sys.version_info[:2] == (3, 11), 'Expected isolated Python 3.11'
assert transformers.__version__ == {expected_tf!r}, 'Transformers version mismatch'
assert torch.__version__.split('+')[0] == {spec['torch']!r}, 'Torch version mismatch'
assert torchvision.__version__.split('+')[0] == {spec['torchvision']!r}, 'Torchvision version mismatch'
if {require_gpu!r}:
    assert torch.cuda.is_available(), 'Select a GPU runtime before running this profile'
if {profile!r} == 'dots':
    import flash_attn
    if {require_gpu!r}:
        assert torch.cuda.get_device_capability()[0] >= 8, 'dots.ocr requires an Ampere-or-newer GPU (L4/A100)'
if {profile!r} == 'chandra':
    from chandra.prompts import PROMPT_MAPPING
    assert 'ocr_layout' in PROMPT_MAPPING
if {profile!r} == 'florence':
    import timm, einops
print(json.dumps({{'python': sys.version.split()[0], 'torch': torch.__version__, 'transformers': transformers.__version__,
                  'tokenizers': tokenizers.__version__, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}))
'''
    result = subprocess.run([str(python), '-c', script], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{profile} environment check failed:\n{result.stderr[-8000:]}')
    return json.loads(result.stdout.strip().splitlines()[-1])


def prepare_environment(profile, env_root, install=True):
    """Linux/Colab setup, no notebook-kernel torch/transformers mutation."""
    if sys.platform != 'linux':
        raise RuntimeError('GPU environment setup targets Linux/Colab; offline results can still be reviewed locally.')
    python, constraint_path, commands = install_commands(profile, env_root)
    if install:
        # uv is the only bootstrap package installed in the notebook interpreter.
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'uv>=0.8,<1'])
        for command in commands[:2]:
            subprocess.check_call(command)
        spec = PROFILES[profile]
        constraint_path.write_text(f"torch=={spec['torch']}\ntorchvision=={spec['torchvision']}\n"
            + '\n'.join(x for x in spec['packages'] if '==' in x) + '\n', encoding='utf-8')
        env = {**os.environ, 'MAX_JOBS': '2', 'NVCC_THREADS': '2'}
        for command in commands[2:]:
            subprocess.check_call(command, env=env)
    if not python.exists():
        raise FileNotFoundError('This profile environment does not exist yet. Set INSTALL_PACKAGES=True.')
    metadata = probe_environment(python, profile)
    print('Verified isolated model environment:', metadata)
    return python
