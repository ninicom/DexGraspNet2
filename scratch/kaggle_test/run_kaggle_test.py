"""Build, verify, train, and export the proven DexGraspNet2 Kaggle GPU env."""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


VENV_DIR = Path("/tmp/dgn_kaggle_env")
BUILD_DIR = Path("/tmp/dgn_kaggle_build")
OUTPUT_DIR = Path("/kaggle/working")
ARCHIVE = OUTPUT_DIR / "dgn_env_built.tar.gz"


def run(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("===> " + " ".join(str(arg) for arg in args), flush=True)
    result = subprocess.run(
        args,
        cwd=cwd,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout, flush=True)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args)
    return result


def ensure_micromamba() -> Path:
    micromamba = BUILD_DIR / "bin/micromamba"
    if micromamba.exists():
        return micromamba
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    run([
        "bash", "-lc",
        f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        f"| tar -xj -C {BUILD_DIR} bin/micromamba",
    ])
    return micromamba


def configure_environment() -> Path:
    python = VENV_DIR / "bin/python"
    if not python.exists():
        micromamba = ensure_micromamba()
        run([
            str(micromamba), "create", "-y", "-p", str(VENV_DIR),
            "-c", "pytorch", "-c", "nvidia/label/cuda-11.7.1", "-c", "conda-forge",
            "python=3.8", "pytorch=2.0.1", "pytorch-cuda=11.7",
            "cuda-nvcc=11.7", "cuda-cudart-dev=11.7",
            "libcublas-dev=11.10.3.66", "libcurand-dev=10.2.10.91",
            "libcusolver-dev=11.4.0.1", "libcusparse-dev=11.7",
            "mkl<2024.1", "mkl-include<2024.1", "pip", "ninja",
        ])
        run([
            str(python), "-m", "pip", "install", "--no-cache-dir",
            "setuptools<70", "numpy==1.23.5", "scipy==1.10.1",
            "PyYAML==6.0.1", "tqdm==4.66.5", "einops==0.7.0",
            "nflows==0.14", "diffusers==0.21.4", "huggingface-hub==0.19.4",
            "easydict", "pillow", "pandas==2.0.3", "plotly==5.24.1",
            "trimesh==4.5.3", "transforms3d==0.4.2", "urdf-parser-py",
        ])

        pytorch3d_package = BUILD_DIR / "pytorch3d-0.7.5-py38_cu117_pyt201.tar.bz2"
        run([
            "curl", "-L", "--fail", "-o", str(pytorch3d_package),
            "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/"
            "linux-64/pytorch3d-0.7.5-py38_cu117_pyt201.tar.bz2",
        ])
        run([
            str(micromamba), "install", "-y", "-p", str(VENV_DIR),
            "-c", "pytorch", "-c", "conda-forge", str(pytorch3d_package),
        ])

    capability = subprocess.check_output([
        str(python), "-c",
        "import torch; assert torch.cuda.is_available(); "
        "print('.'.join(map(str, torch.cuda.get_device_capability(0))))",
    ], text=True).strip()
    major, minor = (int(part) for part in capability.split("."))
    arch = capability if (major, minor) <= (8, 6) else "8.6+PTX"
    os.environ.update({
        "CUDA_HOME": str(VENV_DIR),
        "CUDACXX": str(VENV_DIR / "bin/nvcc"),
        "TORCH_CUDA_ARCH_LIST": arch,
        "MAX_JOBS": "2",
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
    })
    return python


def ensure_minkowski(python: Path) -> None:
    probe = run([
        str(python), "-c",
        "import MinkowskiEngine as ME; print(ME.__version__)",
    ], check=False)
    if probe.returncode == 0:
        return

    source = BUILD_DIR / "MinkowskiEngine"
    if source.exists():
        shutil.rmtree(source)
    run([
        "git", "clone", "--depth", "1", "--branch", "v0.5.4",
        "https://github.com/NVIDIA/MinkowskiEngine.git", str(source),
    ])
    spmm = source / "src/spmm.cu"
    source_text = spmm.read_text(encoding="utf-8")
    anchor = "#include <ATen/cuda/CUDAContext.h>"
    if "#include <ATen/ATen.h>" not in source_text:
        spmm.write_text(
            source_text.replace(anchor, "#include <ATen/ATen.h>\n" + anchor, 1),
            encoding="utf-8",
        )
    run([
        str(python), "setup.py", "install", "--force_cuda", "--blas=mkl",
        f"--blas_include_dirs={VENV_DIR / 'include'}",
    ], cwd=source)


def verify_gpu_stack(python: Path) -> dict[str, str]:
    code = (
        "import json,torch,pytorch3d,MinkowskiEngine as ME;"
        "from pytorch3d.ops import knn_points;"
        "a=torch.zeros((1,2,3),device='cuda');knn_points(a,a);"
        "x=torch.ones((2,3),device='cuda');"
        "c=torch.tensor([[0,0,0,0],[0,1,1,1]],dtype=torch.int32);"
        "s=ME.SparseTensor(x,coordinates=c);assert s.F.is_cuda;"
        "print(json.dumps({'python':__import__('sys').version.split()[0],"
        "'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'gpu':torch.cuda.get_device_name(0),'pytorch3d':pytorch3d.__version__,"
        "'minkowski':ME.__version__}))"
    )
    result = run([str(python), "-c", code])
    return json.loads(result.stdout.strip().splitlines()[-1])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    print("DEXGRASPNET2_KAGGLE_PIPELINE_START", flush=True)
    if not Path("DexGraspNet2").exists():
        run(["git", "clone", "https://github.com/ninicom/DexGraspNet2.git"])
    os.chdir("DexGraspNet2")

    saved_archives = sorted(glob.glob("/kaggle/input/**/dgn_env_built.tar.gz", recursive=True))
    if saved_archives:
        print(f"RESTORING_ENV {saved_archives[0]}", flush=True)
        run(["tar", "-xzf", saved_archives[0], "-C", "/tmp"])

    python = configure_environment()
    ensure_minkowski(python)
    versions = verify_gpu_stack(python)

    run([str(python), "scratch/kaggle_test/generate_mock_data.py"])
    run([str(python), "src/train.py", "--yaml", "scratch/kaggle_test/train_kaggle_test.yaml"])

    print(f"PACKAGING_VERIFIED_ENV {ARCHIVE}", flush=True)
    run(["tar", "-I", "gzip -1", "-cf", str(ARCHIVE), "-C", "/tmp", VENV_DIR.name])
    manifest = {
        "status": "verified",
        "archive": ARCHIVE.name,
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "training_iterations": 5,
        **versions,
    }
    (OUTPUT_DIR / "dgn_env_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("DEXGRASPNET2_KAGGLE_PIPELINE_OK " + json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
