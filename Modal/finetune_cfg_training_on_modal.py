import modal

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
    gpu="T4",
    timeout=7200,
    volumes={"/outputs": volume}
)
def run_cfg_training():
    import os
    import shutil
    import gdown
    from pathlib import Path
    
    # Clone repository
    os.system("git clone https://github.com/tomas-salgado/se3_diffusion")
    os.chdir("se3_diffusion")

    url = "https://drive.google.com/file/d/1Q6GzE3daHtllSCEe40cn8V_Zvi2HFCRJ/view?usp=drive_link"
    output = "p15_ensemble.pdb"
    gdown.download(url=url, output=output, fuzzy=True)
    
    url = "https://drive.google.com/file/d/1GU3uIZPcwmdIwPLeXc2PFifA5OqZzoaf/view?usp=drive_link"
    output = "ar_ensemble.pdb"
    gdown.download(url=url, output=output, fuzzy=True)

    # Run CFG training
    os.system(
        "micromamba run -n se3 python experiments/train_cfg_se3_diffusion.py "
        "model.use_sequence_conditioning=True "
        "model.conditioning_method=cross_attention "
        "model.cfg_dropout_prob=0.1 "
        "model.sequence_embed.embed_dim=1280 "
        "model.sequence_embed.p15_embedding_path=embeddings/p15_idr_embedding.txt "
        "model.sequence_embed.ar_embedding_path=embeddings/ar_idr_embedding.txt "
        "data.p15_data_path=conformations/p15 "
        "data.ar_data_path=conformations/ar "
        "training.batch_size=32 "
        "training.learning_rate=1e-4"
    )
    
    # Copy training outputs back to Modal volume
    if os.path.exists("training_outputs"):
        shutil.copytree("training_outputs", "/outputs/cfg_training_outputs", dirs_exist_ok=True)
    volume.commit()

def main():
    with modal.enable_output():
        with app.run():
            run_cfg_training.remote()

if __name__ == "__main__":
    main()