from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import gc
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)
parser.add_argument("--cpu_offload", action="store_true", help="Offload VAE to CPU to save VRAM")
parser.add_argument("--sequential_decode", action="store_true", help="Decode frames sequentially")
parser.add_argument("--use_fp16", action="store_true", help="Use FP16 precision")

args = parser.parse_args()

torch.set_grad_enabled(False)

# Memory optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(True)

# Clear any existing cache
torch.cuda.empty_cache()
gc.collect()

config = OmegaConf.load(args.config_path)

# Determine dtype
dtype = torch.float16 if args.use_fp16 else torch.bfloat16

pipeline = InferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=dtype)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")['generator']
pipeline.generator.load_state_dict(state_dict, strict=True)

# Store original VAE device for later restoration
original_vae_device = next(pipeline.vae.parameters()).device

# Only move VAE to CPU after inference, not before
if args.cpu_offload:
    print("VAE will be moved to CPU during decoding for memory savings")

dataset = TextDataset(args.prompt_file_path)

sampled_noise = torch.randn([1, 21, 16, 60, 104], device="cuda", dtype=dtype)

os.makedirs(args.output_folder, exist_ok=True)

def sequential_vae_decode(vae, latents, cpu_offload=False):
    """Decode latents frame by frame to save memory"""
    # Process each frame separately
    frames = []
    for i in range(latents.shape[1]):  # iterate over frames
        frame_latent = latents[:, i:i+1]  # [1, 1, C, H, W]
        
        if cpu_offload:
            # Move VAE to CPU and frame latents to CPU
            vae = vae.cpu()
            frame_latent = frame_latent.cpu()
            print(f"Processing frame {i} on CPU")
        else:
            # Keep everything on GPU
            vae = vae.cuda()
            frame_latent = frame_latent.cuda()
            print(f"Processing frame {i} on GPU")
        
        # Decode single frame
        with torch.no_grad():
            frame_pixels = vae.decode_to_pixel(frame_latent)
        
        # Always move result to CPU to save memory
        frame_pixels = frame_pixels.cpu()
        frames.append(frame_pixels)
        
        # Clear cache after each frame
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Concatenate all frames
    result = torch.cat(frames, dim=1)
    
    return result

def safe_vae_decode(vae, latents, cpu_offload=False, sequential=False):
    """Safely decode latents with proper device management"""
    if sequential:
        return sequential_vae_decode(vae, latents, cpu_offload)
    
    if cpu_offload:
        # Move both VAE and latents to CPU
        vae = vae.cpu()
        latents = latents.cpu()
        print("Decoding on CPU for memory savings")
    else:
        # Ensure both are on the same GPU device
        vae = vae.cuda()
        latents = latents.cuda()
        print("Decoding on GPU")
    
    with torch.no_grad():
        video_pixels = vae.decode_to_pixel(latents)
    
    # Always return on CPU to save memory
    return video_pixels.cpu()

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]
    
    # Clear cache before inference
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Processing prompt {prompt_index}: {prompts[0]}")
    print(f"GPU memory before inference: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Ensure VAE is on GPU for inference (restore if it was moved to CPU)
    pipeline.vae = pipeline.vae.cuda()
    
    try:
        # Run inference to get latents (everything on GPU)
        with torch.no_grad():
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True
            )
        
        print(f"GPU memory after generator: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # Clear the video tensor to free memory before VAE decode
        del video
        torch.cuda.empty_cache()
        
        # Decode with memory optimization
        print("Starting VAE decoding...")
        video_pixels = safe_vae_decode(
            pipeline.vae, 
            latents, 
            cpu_offload=args.cpu_offload,
            sequential=args.sequential_decode
        )
        
        print(f"GPU memory after VAE decode: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # Convert to numpy and save
        video = video_pixels[0].permute(0, 2, 3, 1).numpy()
        
        export_to_video(
            video, 
            os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), 
            fps=16
        )
        
        print(f"Saved video {prompt_index} successfully")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM Error for prompt {prompt_index}: {e}")
        print("Try using --cpu_offload and --sequential_decode flags")
        break
    except Exception as e:
        print(f"Error processing prompt {prompt_index}: {e}")
        import traceback
        traceback.print_exc()
        continue
    finally:
        # Cleanup
        if 'latents' in locals() and latents is not None:
            del latents
        if 'video_pixels' in locals():
            del video_pixels
        if 'video' in locals():
            del video
        
        # Restore VAE to GPU for next iteration
        pipeline.vae = pipeline.vae.cuda()
        
        torch.cuda.empty_cache()
        gc.collect()

print("Inference completed!")
