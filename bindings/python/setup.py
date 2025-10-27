"""
Simplified setup.py for HPRLP Python package using CMake
"""
import json
import os
import re
import sys
import subprocess
import glob
import sysconfig
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


MIN_CUDA_VERSION = (12, 4)
MIN_CUDA_VERSION_STR = f"{MIN_CUDA_VERSION[0]}.{MIN_CUDA_VERSION[1]}"


def _version_tuple(major: int, minor: int) -> int:
    return major * 100 + minor


def _is_version_supported(version_tuple):
    if not version_tuple:
        return False
    major, minor, _ = version_tuple
    return _version_tuple(major, minor) >= _version_tuple(*MIN_CUDA_VERSION)


def _detect_cuda_version(cuda_path: str):
    """Return (major, minor, raw_string) for the CUDA install or None."""
    version_file_json = os.path.join(cuda_path, 'version.json')
    version_file_txt = os.path.join(cuda_path, 'version.txt')
    version_text = None

    if os.path.isfile(version_file_json):
        try:
            with open(version_file_json, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            version_text = (
                data.get('cuda', {}).get('version')
                or data.get('version')
                or data.get('cuda_version')
            )
        except Exception:
            version_text = None

    if not version_text and os.path.isfile(version_file_txt):
        try:
            with open(version_file_txt, 'r', encoding='utf-8') as fh:
                contents = fh.read()
            match = re.search(r'CUDA Version\s+([0-9]+)\.([0-9]+)', contents)
            if match:
                version_text = f"{match.group(1)}.{match.group(2)}"
        except Exception:
            version_text = None

    if not version_text:
        match = re.search(r'cuda[-_]?([0-9]+)\.([0-9]+)', cuda_path)
        if match:
            version_text = f"{match.group(1)}.{match.group(2)}"

    if not version_text:
        nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc')
        if os.path.isfile(nvcc_path):
            try:
                output = subprocess.check_output([nvcc_path, '--version'], text=True)
                match = re.search(r'release\s+([0-9]+)\.([0-9]+)', output)
                if match:
                    version_text = f"{match.group(1)}.{match.group(2)}"
            except Exception:
                version_text = None

    if not version_text:
        return None

    try:
        parts = version_text.split('.')
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return major, minor, version_text
    except (ValueError, IndexError):
        return None


def find_cuda_home():
    """Locate a CUDA installation that satisfies the minimum version requirement."""

    def add_candidate(path, source, collection):
        if not path:
            return
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return
        if any(entry['path'] == abs_path for entry in collection):
            return
        collection.append({'path': abs_path, 'source': source})

    candidates = []

    # Priority 1: user-provided environment variables
    add_candidate(os.environ.get('CUDA_HOME'), 'env:CUDA_HOME', candidates)
    add_candidate(os.environ.get('CUDA_PATH'), 'env:CUDA_PATH', candidates)

    # Priority 2: common symlink location
    add_candidate('/usr/local/cuda', 'default', candidates)

    # Priority 3: versioned installs (collect all for evaluation)
    for cuda_dir in sorted(glob.glob('/usr/local/cuda-*'), reverse=True):
        add_candidate(cuda_dir, 'versioned', candidates)

    # Priority 4: other common locations
    for cuda_dir in ('/opt/cuda', '/usr/lib/cuda'):
        add_candidate(cuda_dir, 'fallback', candidates)

    if not candidates:
        raise RuntimeError(
            'CUDA toolkit not found. Install CUDA >= ' + MIN_CUDA_VERSION_STR
        )

    supported = []
    incompatible = []
    invalid = []

    for candidate in candidates:
        path = candidate['path']
        nvcc_path = os.path.join(path, 'bin', 'nvcc')
        if not os.path.isfile(nvcc_path):
            candidate['error'] = 'nvcc not found'
            invalid.append(candidate)
            continue

        version_tuple = _detect_cuda_version(path)
        if not version_tuple:
            candidate['error'] = 'unable to determine CUDA version'
            invalid.append(candidate)
            continue

        major, minor, version_str = version_tuple
        candidate['version'] = (major, minor)
        candidate['version_str'] = version_str

        if _is_version_supported(version_tuple):
            supported.append(candidate)
        else:
            candidate['error'] = (
                f'found CUDA {version_str}, requires >= {MIN_CUDA_VERSION_STR}'
            )
            incompatible.append(candidate)

    if supported:
        # Prefer explicit environment variables first
        for candidate in supported:
            if candidate['source'].startswith('env:'):
                return candidate

        # Otherwise pick the newest version
        supported.sort(
            key=lambda item: _version_tuple(*item['version']), reverse=True
        )
        return supported[0]

    # Construct helpful error message
    messages = []
    for bucket in (incompatible, invalid):
        for candidate in bucket:
            version_info = candidate.get('version_str') or 'unknown version'
            messages.append(
                f"- {candidate['path']} ({version_info}): {candidate.get('error')}"
            )

    detail = '\n'.join(messages) if messages else 'No CUDA toolkits detected.'
    raise RuntimeError(
        'No compatible CUDA installation found. '
        f'HPRLP requires CUDA >= {MIN_CUDA_VERSION_STR}.\n' + detail
    )


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Ensure the extension directory exists
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # CMake configuration arguments
        python_exec = sys.executable
        python_root = sys.prefix or sys.base_prefix

        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_prefix = os.path.abspath(conda_prefix)
            python_exec_real = os.path.abspath(os.path.realpath(python_exec))
            try:
                common = os.path.commonpath([conda_prefix, python_exec_real])
            except ValueError:
                common = ''
            if not common.startswith(conda_prefix):
                suggested_python = os.path.join(conda_prefix, 'bin', 'python')
                raise RuntimeError(
                    'Detected active conda environment at {env} but the build is using {exe} as the Python interpreter.\n'
                    'Please reinstall using the environment-aware interpreter, for example:\n'
                    f'  {suggested_python} -m pip install .\n'
                    'or activate the environment explicitly before running pip.'
                    .format(env=conda_prefix, exe=python_exec_real)
                )

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPython3_EXECUTABLE={python_exec}',
            f'-DPython3_ROOT_DIR={python_root}',
            '-DPython3_FIND_STRATEGY=LOCATION',
            '-DPython3_FIND_VIRTUALENV=FIRST',
            f'-DPYTHON_EXECUTABLE={python_exec}',  # legacy CMake variable for older scripts
            '-DBUILD_PYTHON_BINDINGS=ON',
            '-DBUILD_SHARED_LIB=ON',  # Python bindings need shared library
            '-DBUILD_STATIC_LIB=OFF',  # Don't need static for Python
            '-DBUILD_EXAMPLES=OFF',    # Don't build examples during pip install
        ]

        # Work out Python include directories. Some bare-metal Python installs lack headers unless
        # python-dev is available, so fail early with a helpful hint instead of a cryptic CMake error.
        include_candidates = []
        sys_paths = sysconfig.get_paths()
        for key in ('include', 'platinclude'):
            path = sys_paths.get(key)
            if path:
                include_candidates.append(path)

        for key in ('INCLUDEPY', 'CONFINCLUDEPY', 'INCLUDEDIR'):
            path = sysconfig.get_config_var(key)
            if path:
                include_candidates.append(path)

        py_version_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
        include_candidates.append(os.path.join(python_root, 'include', py_version_tag))
        include_candidates.append(os.path.join(python_root, 'include'))

        validated_include_dirs = []
        for candidate in include_candidates:
            if not candidate:
                continue
            candidate = os.path.abspath(candidate)
            if candidate in validated_include_dirs:
                continue
            if os.path.isfile(os.path.join(candidate, 'Python.h')):
                validated_include_dirs.append(candidate)

        if validated_include_dirs:
            primary_include = validated_include_dirs[0]
            cmake_args.append(f'-DPython3_INCLUDE_DIR={primary_include}')
            cmake_args.append(f'-DPython3_INCLUDE_DIRS={";".join(validated_include_dirs)}')
        else:
            unique_candidates = sorted(set(os.path.abspath(path) for path in include_candidates if path))
            attempted = '\n  '.join(unique_candidates) if unique_candidates else '(none)'
            raise RuntimeError(
                'Python development headers (Python.h) were not found.\n'
                'A C++ extension is required for the HPRLP bindings.\n'
                'Install the appropriate development package for your Python distribution\n'
                '(e.g. "python3-dev" on Debian/Ubuntu, "python-devel" on RHEL/CentOS,\n'
                'or ensure your virtual environment was created from a full CPython build).\n'
                'Paths checked:\n  '
                + attempted
            )

        # Automatically find CUDA if not set
        try:
            cuda_info = find_cuda_home()
            cuda_home = cuda_info['path']
            cuda_version = cuda_info.get('version_str', 'unknown')
            cmake_args.append(f'-DCUDA_TOOLKIT_ROOT_DIR={cuda_home}')
            cmake_args.append(f'-DCUDAToolkit_ROOT={cuda_home}')
            nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
            if os.path.isfile(nvcc_path):
                cmake_args.append(f'-DCMAKE_CUDA_COMPILER={nvcc_path}')
            print(f"✓ Found CUDA {cuda_version} at: {cuda_home}")
            # Also set environment variables for consistency
            os.environ['CUDA_HOME'] = cuda_home
            os.environ['CUDA_PATH'] = cuda_home
            os.environ['CUDAToolkit_ROOT'] = cuda_home

            # Ensure nvcc and libraries from the selected toolkit are discovered first
            bin_dir = os.path.join(cuda_home, 'bin')
            if os.path.isdir(bin_dir):
                path_env = os.environ.get('PATH', '')
                os.environ['PATH'] = os.pathsep.join(
                    [bin_dir, path_env] if path_env else [bin_dir]
                )

            ld_dirs = [os.path.join(cuda_home, 'lib64'), os.path.join(cuda_home, 'lib')]
            existing_ld = os.environ.get('LD_LIBRARY_PATH', '')
            for ld_dir in ld_dirs:
                if os.path.isdir(ld_dir):
                    os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(
                        [ld_dir, existing_ld] if existing_ld else [ld_dir]
                    )
                    break
        except RuntimeError as err:
            print(f"✗ {err}")
            raise

        # Build configuration
        cfg = 'Release'
        build_args = ['--config', cfg]

        # Platform-specific arguments
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        
        # Use parallel build
        build_args += ['--', '-j4']

        # Create build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake configuration
        print(f"Running CMake configuration in {self.build_temp}")
        
        # Source directory is two levels up (../../)
        source_dir = os.path.abspath(os.path.join(ext.sourcedir, '../..'))
        
        subprocess.check_call(
            ['cmake', source_dir] + cmake_args,
            cwd=self.build_temp
        )
        
        # Run CMake build
        print("Building with CMake...")
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )


# Read README for long description
readme_path = Path(__file__).parent.parent.parent / 'README.md'
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''

setup(
    name='hprlp',
    version='0.1.0',
    author='HPR-LP Contributors',
    description='Python bindings for the GPU-accelerated Halpern–Peaceman–Rachford linear programming solver',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PolyU-IOR/HPR-LP-C',
    ext_modules=[CMakeExtension('hprlp._hprlp_core')],
    cmdclass={'build_ext': CMakeBuild},
    packages=['hprlp'],
    package_dir={'hprlp': 'hprlp'},
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    zip_safe=False,
)
