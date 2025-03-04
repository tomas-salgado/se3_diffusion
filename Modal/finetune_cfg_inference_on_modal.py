import modal

app = modal.App("se3-diffusion-cfg-inference")
volume = modal.Volume.from_name("se3-outputs", create_if_missing=True)

image = modal.Image.micromamba()
image = (
    image
    .apt_install("git")
    .pip_install("gdown")
    .run_commands(
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
def run_cfg_inference():
    import os
    import shutil
    import gdown
    os.system("git clone https://github.com/tomas-salgado/se3_diffusion")
    os.chdir("se3_diffusion")

    url = "https://drive.google.com/file/d/1Q6GzE3daHtllSCEe40cn8V_Zvi2HFCRJ/view?usp=drive_link"
    output = "p15_ensemble.pdb"
    gdown.download(url=url, output=output, fuzzy=True)

    url = "https://drive.google.com/file/d/1GU3uIZPcwmdIwPLeXc2PFifA5OqZzoaf/view?usp=drive_link"
    output = "ar_ensemble.pdb"
    gdown.download(url=url, output=output, fuzzy=True)

    # Copy trained model and embeddings
    os.makedirs("weights", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    
    shutil.copy2("/outputs/cfg_training_outputs/model.pth", "weights/cfg_model.pth")
    shutil.copy2("/outputs/embeddings/p15_embeddings.txt", "embeddings/p15_embeddings.txt")
    shutil.copy2("/outputs/embeddings/ar_embeddings.txt", "embeddings/ar_embeddings.txt")
    
    # Run inference with different CFG scales
    os.system(
        "micromamba run -n se3 python experiments/finetune_inference_se3_diffusion.py "
        "model.use_sequence_conditioning=True "
        "model.conditioning_method=cross_attention "
        "model.sequence_embed.embed_dim=1280 "
        "model.sequence_embed.p15_embedding_path=embeddings/p15_idr_embedding.txt "
        "model.sequence_embed.ar_embedding_path=embeddings/ar_idr_embedding.txt "
        "inference.cfg_scale=3.0 "  # Can adjust this
        "inference.embedding_type=p15"  # or 'ar'
    )
    
    # Copy results back to volume
    if os.path.exists("inference_outputs"):
        shutil.copytree("inference_outputs", "/outputs/cfg_inference_outputs", dirs_exist_ok=True)
    volume.commit()

def main():
    with modal.enable_output():
        with app.run():
            run_cfg_inference.remote()

if __name__ == "__main__":
    main() 