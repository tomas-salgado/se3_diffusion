import modal
import time

app = modal.App("se3-diffusion-cfg")
volume = modal.Volume.from_name("se3-outputs", create_if_missing=True)

# Create base image
image = modal.Image.micromamba()
image = (
    image
    .apt_install("git")
    .pip_install(["gdown"]) 
    .run_commands(
        "git clone https://github.com/tomas-salgado/se3_diffusion",
        "cd se3_diffusion && micromamba env create -f se3.yml",
        "cd se3_diffusion && micromamba run -n se3 pip install -e ."
    )
)

@app.function(
    image=image,
    gpu="A100:2",
    timeout=86400,  # Set timeout to 24 hours (in seconds)
    volumes={"/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def run_cfg_training():
    import os
    import shutil
    import gdown
    import threading
    import time
    from pathlib import Path
    
    # Clone repository
    os.system("git clone https://github.com/tomas-salgado/se3_diffusion")
    os.chdir("se3_diffusion")
    
    # Debug: Print current directory and contents
    print("\nCurrent directory:", os.getcwd())
    print("\nContents of current directory:")
    os.system("ls -la")
    
    print("\nContents of config directory:")
    os.system("ls -la config/")
    
    print("\nChecking for finetune_cfg.yaml specifically:")
    os.system("find . -name 'finetune_cfg.yaml'")
    
    print("\nChecking all yaml files in config:")
    os.system("find config/ -name '*.yaml' -o -name '*.yml'")
    
    # Add current directory to PYTHONPATH
    os.environ['PYTHONPATH'] = os.getcwd() + ":" + os.environ.get('PYTHONPATH', '')

    os.makedirs("ensembles", exist_ok=True)
    
    url = "https://drive.google.com/file/d/1Q6GzE3daHtllSCEe40cn8V_Zvi2HFCRJ/view?usp=drive_link"
    output = "ensembles/p15_ensemble.pdb"
    gdown.download(url=url, output=output, fuzzy=True)
    
    url = "https://drive.google.com/file/d/1GU3uIZPcwmdIwPLeXc2PFifA5OqZzoaf/view?usp=drive_link"
    output = "ensembles/ar_ensemble.pdb"
    gdown.download(url=url, output=output, fuzzy=True)

    # Create checkpoint directory to ensure it exists
    os.makedirs("ckpt", exist_ok=True)
    os.makedirs("eval_outputs", exist_ok=True)
    
    # Function to periodically sync outputs to volume
    def sync_outputs():
        while True:
            time.sleep(30)  # Sync every 5 minutes
            print("\nSyncing outputs to persistent volume...")
            paths_to_save = ["ckpt/", "eval_outputs/", "training_outputs/"]
            for path in paths_to_save:
                if os.path.exists(path):
                    dst = os.path.join("/outputs", path)
                    print(f"Syncing {path} to {dst}")
                    if os.path.isdir(path):
                        shutil.copytree(path, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(path, dst)
            volume.commit()
            print("Sync complete!")
    
    # Start sync thread
    sync_thread = threading.Thread(target=sync_outputs, daemon=True)
    sync_thread.start()
    
    # Run CFG training with more frequent logging and checkpoints
    print("\nStarting training with more frequent checkpoints and logging...")
    os.system(
        "micromamba run -n se3 python experiments/train_cfg_se3_diffusion.py "
    )
    
    # Final sync of outputs to volume
    print("\nPerforming final sync to persistent volume...")
    paths_to_save = ["ckpt/", "eval_outputs/", "training_outputs/"]
    for path in paths_to_save:
        if os.path.exists(path):
            dst = os.path.join("/outputs", path)
            print(f"Final sync of {path} to {dst}")
            if os.path.isdir(path):
                shutil.copytree(path, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(path, dst)
    
    # Check what we're saving to the volume
    print("\nContents of Modal volume /outputs directory:")
    os.system("ls -la /outputs")
    
    # Check specific checkpoint contents
    print("\nDetail of checkpoint files (if any):")
    os.system("find /outputs/ckpt -type f -name '*.pth' | sort")
    
    volume.commit()
    print("\nVolume committed. Training complete!")

def main():
    with modal.enable_output():
        with app.run():
            run_cfg_training.remote()

if __name__ == "__main__":
    main()