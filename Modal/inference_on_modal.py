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
    gpu="A100",
    timeout=3600,
    volumes={"/outputs": volume}
)
def run_inference():
    import os
    import shutil
    from pathlib import Path
    import gdown
    
    # Remove cached repo if it exists
    repo_path = Path("se3_diffusion")
    if repo_path.exists():
        shutil.rmtree(repo_path)

    # Clone repository again for function execution
    os.system("git clone https://github.com/tomas-salgado/se3_diffusion")
    
    # Run inference script
    os.chdir("se3_diffusion")

    os.system("micromamba run -n se3 python experiments/inference_se3_diffusion.py")
    
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