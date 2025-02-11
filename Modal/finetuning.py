import modal
import time

app = modal.App("se3-diffusion-finetune")
volume = modal.Volume.from_name("se3-outputs", create_if_missing=True)

# Create base image
image = modal.Image.micromamba()
image = (
    image
    .apt_install("git", "wget")
    .pip_install("gdown")  
    .run_commands(
        # Add timestamp as comment to force rebuild
        "echo '# Build timestamp: {}' > /dev/null".format(int(time.time())),
        "git clone https://github.com/tomas-salgado/se3_diffusion",
        "cd se3_diffusion && micromamba env create -f se3.yml",
        "cd se3_diffusion && micromamba run -n se3 pip install -e ."
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
    
    # Create weights directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)
    
    # Download MD data and weights from Google Drive
    import gdown
    
    # Download MD data
    md_url = "https://drive.google.com/file/d/1AwNll554qxREWW__Z40sYJf3n2sq2LlI/view"
    md_output = "Tau5R2R3_backbone.npz"
    gdown.download(url=md_url, output=md_output, fuzzy=True)
    
    # Download weights file
    weights_url = "https://drive.google.com/file/d/1fUel-CmAz9G_999vcD9g93EXTysvLfKY/view"
    weights_output = "weights/ar_finetuning.pth"
    gdown.download(url=weights_url, output=weights_output, fuzzy=True)
    
    # Create directories for outputs
    os.makedirs("ckpt", exist_ok=True)
    os.makedirs("eval_outputs", exist_ok=True)
    
    # Function to periodically sync outputs to volume
    def sync_outputs():
        while True:
            time.sleep(300)  # Sync every 5 minutes
            print("\nSyncing outputs to persistent volume...")
            paths_to_save = ["ckpt/", "eval_outputs/", "outputs/", "weights/"]  # Added weights/ to sync
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
        "experiment.ckpt_epochs=1 "  # Save checkpoint every epoch
        "experiment.eval_epochs=2"    # Evaluate every 2 epochs
    )
    
    # Final sync of outputs to volume
    print("\nPerforming final sync to persistent volume...")
    paths_to_save = ["ckpt/", "eval_outputs/", "outputs/", "weights/"]  # Added weights/ to sync
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