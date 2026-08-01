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
    print("Starting Kaggle GPU Test for DexGraspNet2...")

    # 1. Clone repository
    if not os.path.exists("DexGraspNet2"):
        run_cmd("git clone https://github.com/ninicom/DexGraspNet2.git")

    os.chdir("DexGraspNet2")

    # 2. Install lightweight dependencies
    print("Installing python requirements...")
    run_cmd("pip install --quiet easydict scipy Pillow pyyaml tqdm einops pytorch3d")

    # 3. Copy scratch test scripts into local workspace if needed
    os.makedirs("scratch/kaggle_test", exist_ok=True)
    
    # 4. Generate mock dataset
    print("Generating 100% schema-compliant synthetic mock dataset...")
    run_cmd("python3 scratch/kaggle_test/generate_mock_data.py")

    # 5. Execute trial training
    print("Executing trial training on Kaggle GPU (cuda:0)...")
    run_cmd("python3 src/train.py --yaml scratch/kaggle_test/train_kaggle_test.yaml")
    print("Kaggle GPU Trial Training completed successfully!")

if __name__ == "__main__":
    main()
