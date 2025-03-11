import modal
import time

app = modal.App("se3-diffusion-cfg-inference")
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
    timeout=3600,  # 1 hour
    volumes={"/outputs": volume}
)
def run_cfg_inference():
    import os
    import shutil
    import gdown
    from pathlib import Path
    
    # Clone repository
    os.system("git clone https://github.com/tomas-salgado/se3_diffusion")
    os.chdir("se3_diffusion")
    
    # Create directories
    os.makedirs("weights", exist_ok=True)
    
    # Copy weights from Modal storage volume to local directory
    weights_src = "/outputs/ckpt/baseline/10D_03M_2025Y_17h_22m_40s/step_5000.pth"
    weights_dst = "weights/checkpoint.pth"
    if os.path.exists(weights_src):
        shutil.copy2(weights_src, weights_dst)
        print(f"Successfully loaded weights from Modal volume at {weights_src}")
    else:
        raise FileNotFoundError(f"Weights file not found in Modal volume at {weights_src}")
    
    # Create output directory for generated structures
    output_base_dir = "cfg_inference_outputs"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Define guidance scales to test
    cfg_scales = [1.0, 3.0, 5.0, 7.5, 10.0]
    
    # Run inference for each embedding and guidance scale
    embeddings = [
        ("embeddings/p15PAF_idr_embedding.txt", "p15"),
        ("embeddings/ar_idr_embedding.txt", "ar")
    ]
    
    for embedding_path, name in embeddings:
        for cfg_scale in cfg_scales:
            print(f"\nRunning inference for {name} with CFG scale {cfg_scale}")
            
            # Create output directory for this run
            output_dir = os.path.join(output_base_dir, f"{name}_cfg{cfg_scale:.1f}")
            
            # Run inference with the new CFG inference script, now using local weights path
            cmd = (
                f"cd $(pwd) && micromamba run -n se3 "
                f"python -m experiments.inference_cfg_se3_diffusion "
                f"embedding_path={embedding_path} "
                f"output_dir={output_dir} "
                f"cfg_scale={cfg_scale} "
                f"num_samples=5"
            )
            print(f"Executing: {cmd}")
            os.system(cmd)
    
    # Move results to mounted volume
    print("\nCopying inference results to Modal volume...")
    
    # Create destination directory in the mounted volume
    os.makedirs("/outputs/cfg_inference_outputs", exist_ok=True)
    
    # Copy each result directory individually to ensure proper organization
    for embedding_path, name in embeddings:
        for cfg_scale in cfg_scales:
            src_dir = os.path.join(output_base_dir, f"{name}_cfg{cfg_scale:.1f}")
            dst_dir = os.path.join("/outputs/cfg_inference_outputs", f"{name}_cfg{cfg_scale:.1f}")
            
            if os.path.exists(src_dir):
                print(f"Copying {src_dir} to {dst_dir}")
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    
    # List the generated files
    print("\nGenerated files in Modal volume:")
    os.system("find /outputs/cfg_inference_outputs -name '*.pdb' | sort")
    
    # Commit changes to the volume
    volume.commit()
    print("\nInference complete! All results saved to Modal volume.")

def main():
    with modal.enable_output():
        with app.run():
            run_cfg_inference.remote()

if __name__ == "__main__":
    main() 