import subprocess
import sys
import os
import glob

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

    venv_dir = "/tmp/dgn_kaggle_env"
    python_bin = f"{venv_dir}/bin/python"

    # 2. Check if pre-saved environment tarball exists from attached dataset
    saved_env_tar = None
    input_tars = glob.glob("/kaggle/input/**/dgn_env*.tar.gz", recursive=True)
    if input_tars:
        saved_env_tar = input_tars[0]

    if saved_env_tar and os.path.exists(saved_env_tar):
        print(f"\n[1/5] Found pre-saved environment archive at {saved_env_tar}!")
        print("Extracting environment in seconds (offline instant setup)...")
        run_cmd(f"tar -xzf {saved_env_tar} -C /tmp/")
    else:
        print(f"\n[1/5] Setting up dedicated virtual environment at {venv_dir}...")
        run_cmd("pip install --quiet virtualenv")
        
        py310_path = subprocess.getoutput("which python3.10 || which python3.11").strip()
        if py310_path and os.path.exists(py310_path):
            print(f"Found Python binary: {py310_path}")
            run_cmd(f"virtualenv --python={py310_path} --quiet {venv_dir}")
        else:
            run_cmd(f"virtualenv --quiet {venv_dir}")

        pip_bin = f"{venv_dir}/bin/pip"
        py_ver = subprocess.getoutput(f"{python_bin} -c \"import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')\"").strip()
        print(f"Virtual environment Python version: {py_ver}")

        # Check attached dataset for pre-downloaded wheels
        dataset_wheels_dir = "/kaggle/input/dgn-env-wheels"
        if not os.path.exists(dataset_wheels_dir):
            matches = glob.glob("/kaggle/input/**/dgn-env-wheels*", recursive=True)
            if matches:
                dataset_wheels_dir = matches[0]

        print(f"Found attached wheel dataset at: {dataset_wheels_dir}")

        # Install wheels instantly from attached dataset
        print("\n[2/5] Installing packages from attached wheel dataset...")
        run_cmd(f"{pip_bin} install --quiet --upgrade pip setuptools wheel wrapt")
        run_cmd(f"{pip_bin} install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        run_cmd(f"{pip_bin} install --quiet --find-links {dataset_wheels_dir} easydict scipy Pillow pyyaml tqdm einops fvcore iopath pytorch3d")

        # Install MinkowskiEngine
        print("\n[3/5] Installing MinkowskiEngine from source on Kaggle GPU...")
        run_cmd("apt-get update -y && apt-get install -y libopenblas-dev build-essential", check=False)
        if not os.path.exists("/tmp/MinkowskiEngine"):
            run_cmd("git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine")
        
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        os.environ["MAX_JOBS"] = "4"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5 8.0 8.6"
        
        run_cmd(f"cd /tmp/MinkowskiEngine && {python_bin} setup.py install --blas=openblas --force_cuda")

        # Save compiled environment to Kaggle output for instant reuse
        print("\nSaving compiled environment to /kaggle/working/dgn_env_built.tar.gz for instant reuse...")
        run_cmd("tar -czf /kaggle/working/dgn_env_built.tar.gz -C /tmp dgn_kaggle_env")

    # 3. Generate mock dataset
    print("\n[4/5] Generating 100% schema-compliant synthetic mock dataset directly on Kaggle...")
    run_cmd(f"{python_bin} scratch/kaggle_test/generate_mock_data.py")

    # 4. Execute trial training on Kaggle GPU
    print("\n[5/5] Executing trial training on Kaggle GPU (cuda:0)...")
    run_cmd(f"{python_bin} src/train.py --yaml scratch/kaggle_test/train_kaggle_test.yaml")

    print("\n=========================================================")
    print("   KAGGLE DIRECT GPU PIPELINE TEST COMPLETED SUCCESSFULLY! ")
    print("=========================================================")

if __name__ == "__main__":
    main()
