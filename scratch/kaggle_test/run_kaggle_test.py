import subprocess
import sys
import os

def run_cmd(cmd):
    print(f"===> Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}")
        sys.exit(res.returncode)

def main():
    print("Starting Kaggle GPU Test with isolated custom virtual environment...")

    # 1. Clone repository
    if not os.path.exists("DexGraspNet2"):
        run_cmd("git clone https://github.com/ninicom/DexGraspNet2.git")

    os.chdir("DexGraspNet2")

    # 2. Create isolated virtual environment using virtualenv
    venv_dir = "/tmp/custom_venv"
    print(f"Creating isolated virtual environment at {venv_dir}...")
    run_cmd("pip install --quiet virtualenv wrapt")
    run_cmd(f"virtualenv --quiet {venv_dir}")

    pip_bin = f"{venv_dir}/bin/pip"
    python_bin = f"{venv_dir}/bin/python"

    # 3. Upgrade base packaging tools
    print("Upgrading pip, setuptools, wheel in virtual environment...")
    run_cmd(f"{pip_bin} install --quiet --upgrade pip setuptools wheel wrapt")

    # 4. Install dependencies in virtual environment
    print("Installing PyTorch and dependencies into custom virtual environment...")
    run_cmd(f"{pip_bin} install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    run_cmd(f"{pip_bin} install --quiet easydict scipy Pillow pyyaml tqdm einops fvcore iopath")
    
    print("Installing PyTorch3D prebuilt wheel into custom virtual environment...")
    # Try installing prebuilt wheel
    res = subprocess.run(f"{pip_bin} install --quiet pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("Prebuilt wheel 1 failed, trying fallback index...")
        run_cmd(f"{pip_bin} install --quiet pytorch3d --extra-index-url https://miovision.github.io/pytorch3d-wheels/cu121/")

    # 5. Generate mock dataset
    print("Generating 100% schema-compliant synthetic mock dataset...")
    run_cmd(f"{python_bin} scratch/kaggle_test/generate_mock_data.py")

    # 6. Execute trial training using isolated virtual environment python
    print("Executing trial training on Kaggle GPU using custom virtual environment...")
    run_cmd(f"{python_bin} src/train.py --yaml scratch/kaggle_test/train_kaggle_test.yaml")
    print("Kaggle GPU Trial Training in custom venv completed successfully!")

if __name__ == "__main__":
    main()
