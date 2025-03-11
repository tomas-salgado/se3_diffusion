import modal

app = modal.App("se3-diffusion")
volume = modal.Volume.from_name("se3-outputs", create_if_missing=True)

# Create base image
image = modal.Image.micromamba()
image = (
    image
    .apt_install("git")
    .pip_install("gdown")
    .run_commands(
        # Clone the repository
        "git clone https://github.com/tomas-salgado/se3_diffusion",
        # Create conda environment from yml file
        "cd se3_diffusion && micromamba env create -f se3.yml",
        # Install the package in editable mode
        "cd se3_diffusion && micromamba run -n se3 pip install -e ."
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/outputs": volume}
)
def run_inference():
    import os
    import shutil
    from pathlib import Path
    import gdown
    
    # Clone repository again for function execution
    os.system("git clone https://github.com/tomas-salgado/se3_diffusion")
    
    # Run inference script
    os.chdir("se3_diffusion")

    os.makedirs("weights", exist_ok=True)
    # Copy weights from Modal storage volume to local directory in Modal container
    weights_src = "/outputs/ckpt/ar_finetune/26D_02M_2025Y_22h_18m_09s/step_60000.pth"
    weights_dst = "weights/ar_finetuning.pth"
    if os.path.exists(weights_src):
        shutil.copy2(weights_src, weights_dst)
        print(f"Successfully loaded weights from Modal volume")
    else:
        raise FileNotFoundError(f"Weights file not found in Modal volume at {weights_src}")

    # Run inference with sequence conditioning and CFG
    os.system(
        "micromamba run -n se3 python experiments/finetune_inference_se3_diffusion.py "
        "model.use_sequence_conditioning=True "
        "model.conditioning_method=cross_attention "
        "model.sequence_embed.embed_dim=256 "
        "model.sequence_embed.embedding_path=embeddings/p15PAF_idr_embeddings.txt "
        "model.sequence_embed.embedding_format=txt "
        "inference.cfg_scale=3.0"  # Add CFG scale parameter
    )
    
    # Move results to mounted volume and commit
    if os.path.exists("inference_outputs"):
        for item in os.listdir("inference_outputs"):
            src = os.path.join("inference_outputs", item)
            dst = os.path.join("/outputs", item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        volume.commit()

def main():
    with modal.enable_output():
        with app.run():
            run_inference.remote()

if __name__ == "__main__":
    main()