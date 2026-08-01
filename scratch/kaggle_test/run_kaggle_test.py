import subprocess
import sys
import os
import glob

def run_cmd(cmd):
    print(f"===> Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}")
        sys.exit(res.returncode)

def main():
    print("Starting Kaggle GPU Test with offline dataset environment (dgn-env-wheels)...")

    # 1. Clone repository
    if not os.path.exists("DexGraspNet2"):
        run_cmd("git clone https://github.com/ninicom/DexGraspNet2.git")

    os.chdir("DexGraspNet2")

    # 2. Locate Kaggle offline wheels dataset
    dataset_wheels_dir = "/kaggle/input/dgn-env-wheels"
    if not os.path.exists(dataset_wheels_dir):
        print(f"Warning: {dataset_wheels_dir} not found directly, searching /kaggle/input...")
        matches = glob.glob("/kaggle/input/**/dgn-env-wheels*", recursive=True)
        if matches:
            dataset_wheels_dir = matches[0]

    print(f"Found offline wheel dataset at: {dataset_wheels_dir}")

    # 3. Create isolated virtual environment
    venv_dir = "/tmp/dgn_venv"
    print(f"Creating isolated virtual environment at {venv_dir}...")
    run_cmd("pip install --quiet virtualenv wrapt")
    run_cmd(f"virtualenv --quiet {venv_dir}")

    pip_bin = f"{venv_dir}/bin/pip"
    python_bin = f"{venv_dir}/bin/python"

    # 4. Install dependencies offline from attached Kaggle Dataset
    print(f"Installing wheels offline from {dataset_wheels_dir}...")
    run_cmd(f"{pip_bin} install --no-index --find-links {dataset_wheels_dir} easydict scipy Pillow pyyaml tqdm einops fvcore iopath wrapt pytorch3d || {pip_bin} install --find-links {dataset_wheels_dir} easydict scipy Pillow pyyaml tqdm einops fvcore iopath wrapt pytorch3d")

    # 5. Generate mock dataset
    print("Generating 100% schema-compliant synthetic mock dataset...")
    run_cmd(f"{python_bin} scratch/kaggle_test/generate_mock_data.py")

    # 6. Execute trial training using isolated virtual environment python
    print("Executing trial training on Kaggle GPU using custom virtual environment...")
    run_cmd(f"{python_bin} src/train.py --yaml scratch/kaggle_test/train_kaggle_test.yaml")
    print("Kaggle GPU Trial Training in offline dataset venv completed successfully!")

if __name__ == "__main__":
    main()
