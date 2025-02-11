import modal
import time

app = modal.App("se3-diffusion-finetune")
volume = modal.Volume.from_name("se3-outputs", create_if_missing=True)

# Create base image
image = modal.Image.micromamba()
image = (
    image
    .apt_install("git", "wget")
    .pip_install(
        "gdown",
        "antlr4-python3-runtime==4.9.3",
        "biopython==1.80",
        "blessed==1.20.0",
        "click==8.1.3",
        "contextlib2==21.6.0",
        "deepspeed==0.8.0",
        "docker-pycreds==0.4.0",
        "fair-esm==2.0.0",
        "gitdb==4.0.10",
        "gitpython==3.1.30",
        "gpustat==1.0.0",
        "gputil==1.4.0",
        "hjson==3.1.0",
        "hydra-core==1.3.1",
        "hydra-joblib-launcher==1.2.0",
        "joblib==1.2.0",
        "ml-collections==0.1.1",
        "ninja==1.11.1",
        "numpy==1.22.4",
        "nvidia-ml-py==11.495.46",
        "omegaconf==2.3.0",
        "pathtools==0.1.2",
        "protobuf==4.21.12",
        "py-cpuinfo==9.0.0",
        "pydantic==1.10.4",
        "scikit-learn==1.2.1",
        "sentry-sdk==1.15.0",
        "setproctitle==1.3.2",
        "smmap==5.0.0",
        "threadpoolctl==3.1.0",
        "tmtools==0.0.2",
        "tqdm==4.64.1",
        "wandb==0.13.10"
    )
    .run_commands(
        # Add timestamp as comment to force rebuild
        "echo '# Build timestamp: {}' > /dev/null".format(int(time.time())),
        "git clone https://github.com/tomas-salgado/se3_diffusion",
        # Install in development mode
        "cd se3_diffusion && pip install -e ."
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/outputs": volume}
)
def run_finetuning():
    import os
    import shutil
    import threading
    import time
    
    # Remove any existing repo directory
    os.system("rm -rf se3_diffusion")
    
    # Clone fresh copy of repository and set up
    os.system("git clone https://github.com/tomas-salgado/se3_diffusion")
    os.chdir("se3_diffusion")
    
    # Download MD data from Google Drive
    import gdown
    url = "https://drive.google.com/file/d/1AwNll554qxREWW__Z40sYJf3n2sq2LlI/view"
    output = "Tau5R2R3_backbone.npz"
    gdown.download(url=url, output=output, fuzzy=True)
    
    # Create directories for outputs
    os.makedirs("ckpt", exist_ok=True)
    os.makedirs("eval_outputs", exist_ok=True)
    
    # Function to periodically sync outputs to volume
    def sync_outputs():
        while True:
            time.sleep(300)  # Sync every 5 minutes
            print("\nSyncing outputs to persistent volume...")
            paths_to_save = ["ckpt/", "eval_outputs/", "outputs/"]
            for path in paths_to_save:
                if os.path.exists(path):
                    dst = os.path.join("/outputs", path)
                    if os.path.isdir(path):
                        shutil.copytree(path, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(path, dst)
            volume.commit()
            print("Sync complete!")
    
    # Start sync thread
    sync_thread = threading.Thread(target=sync_outputs, daemon=True)
    sync_thread.start()
    
    # Run fine-tuning with more frequent checkpoints
    os.system(
        "micromamba run -n se3 python experiments/finetune_ar.py "
        "experiment.use_wandb=False "
        "experiment.ckpt_freq=1000 "  # Save checkpoint every 1000 steps
        "experiment.eval_freq=2000"    # Evaluate every 2000 steps
    )
    
    # Final sync of outputs to volume
    print("\nPerforming final sync to persistent volume...")
    paths_to_save = ["ckpt/", "eval_outputs/", "outputs/"]
    for path in paths_to_save:
        if os.path.exists(path):
            dst = os.path.join("/outputs", path)
            if os.path.isdir(path):
                shutil.copytree(path, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(path, dst)
    
    volume.commit()
    print("Final sync complete!")

def main():
    with modal.enable_output():
        with app.run():
            run_finetuning.remote()

if __name__ == "__main__":
    main()