import subprocess
import sys
import os

def run_cmd(cmd, check=True):
    print(f"===> Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if check and res.returncode != 0:
        print(f"Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    return res

def main():
    print("=========================================================")
    print("      DEXGRASPNET2 KAGGLE DIRECT GPU PIPELINE TEST       ")
    print("=========================================================")

    # 1. Clone repository on Kaggle
    if not os.path.exists("DexGraspNet2"):
        run_cmd("git clone https://github.com/ninicom/DexGraspNet2.git")
    os.chdir("DexGraspNet2")

    # 2. Set up isolated virtual environment directly on Kaggle
    venv_dir = "/tmp/dgn_kaggle_env"
    print(f"\n[1/5] Setting up dedicated virtual environment at {venv_dir}...")
    run_cmd("pip install --quiet virtualenv")
    run_cmd(f"virtualenv --quiet {venv_dir}")

    pip_bin = f"{venv_dir}/bin/pip"
    python_bin = f"{venv_dir}/bin/python"

    print(f"Virtual environment Python version: {subprocess.getoutput(f'{python_bin} --version')}")

    # 3. Install PyTorch & dependencies directly on Kaggle
    print("\n[2/5] Installing PyTorch and dependencies into Kaggle venv...")
    run_cmd(f"{pip_bin} install --quiet --upgrade pip setuptools wheel wrapt")
    run_cmd(f"{pip_bin} install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    run_cmd(f"{pip_bin} install --quiet easydict scipy Pillow pyyaml tqdm einops fvcore iopath")

    # 4. Install PyTorch3D directly on Kaggle
    print("\n[3/5] Installing PyTorch3D into Kaggle venv...")
    # Try prebuilt wheel link first, then source fallback
    p3d_res = run_cmd(f"{pip_bin} install --quiet pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html", check=False)
    if p3d_res.returncode != 0:
        print("Prebuilt PyTorch3D wheel not found for current Python/CUDA version, building from PyTorch3D git source...")
        run_cmd(f"{pip_bin} install --quiet 'git+https://github.com/facebookresearch/pytorch3d.git'")

    # 5. Generate mock dataset directly on Kaggle
    print("\n[4/5] Generating 100% schema-compliant synthetic mock dataset directly on Kaggle...")
    run_cmd(f"{python_bin} scratch/kaggle_test/generate_mock_data.py")

    # 6. Execute trial training on Kaggle GPU
    print("\n[5/5] Executing trial training on Kaggle GPU (cuda:0)...")
    run_cmd(f"{python_bin} src/train.py --yaml scratch/kaggle_test/train_kaggle_test.yaml")

    print("\n=========================================================")
    print("   KAGGLE DIRECT GPU PIPELINE TEST COMPLETED SUCCESSFULLY! ")
    print("=========================================================")

if __name__ == "__main__":
    main()
