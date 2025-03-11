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
    gpu="A100:2",
    volumes={"/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=86400  # Set timeout to 24 hours (in seconds)
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
    
    # Download data from Google Drive
    import gdown
    url = "https://drive.google.com/file/d/1uNeCkgqEe_qNhoAnsE3faZmfTtGrAoMe/view?usp=drive_link"
    output = "PED00016e001.pdb"
    gdown.download(url=url, output=output, fuzzy=True)

    # url = "https://drive.google.com/file/d/1YLJqThMT-4vhI-ArEnmOvc-FaAE9zB1F/view?usp=drive_link"
    # output = "Tau5R2R3_apo.xtc"
    # gdown.download(url=url, output=output, fuzzy=True)

    # url = "https://drive.google.com/file/d/1Y_bkr-YHdgvF4S3Cuxh-Qa2-AVwdpH_W/view?usp=drive_link"
    # output = "Tau5R2R3_apo.pdb"
    # gdown.download(url=url, output=output, fuzzy=True)

    # # Download sequence embeddings for IDRs
    # os.makedirs("embeddings", exist_ok=True)
    # url = "YOUR_EMBEDDINGS_URL_HERE"  # Replace with your embeddings URL
    # output = "embeddings/idr_embeddings.pkl"
    # gdown.download(url=url, output=output, fuzzy=True)

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
    
    # Run fine-tuning with sequence conditioning
    os.system(
        "micromamba run -n se3 python experiments/finetune_se3_diffusion.py "
        "experiment.use_wandb=True "
        "experiment.ckpt_epochs=1 "
        "experiment.eval_epochs=2 "
        "model.use_sequence_conditioning=True "
        "model.conditioning_method=cross_attention "
        "model.sequence_embed.embed_dim=256 "  # Make sure this matches your embedding dimension
        "model.sequence_embed.embedding_path=embeddings/p15PAF_idr_embedding.txt"  # Your text file path
        "model.sequence_embed.embedding_format=txt"  # Specify text format
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