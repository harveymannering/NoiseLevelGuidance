import torch
import random
import os
from tqdm import tqdm
from utils import load_generation_arguments, set_seed, load_stable_diffusion, load_ms_coco_prompts, refine_noise, save_images_from_numpy
import pandas as pd

def main():
    
    # Are we using the CPU or the GPU?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Device:', device, flush=True)
    
    # Load command line arguments
    args = load_generation_arguments()
    
    # Set random seed for reproducible results
    set_seed(0)
    
    # Set up the Stable Diffusion model
    stable_diffusion = load_stable_diffusion(args.stable_diffusion_path, device)
    
    # Load and shuffle prompts to be used for generation
    prompts = load_ms_coco_prompts(args.prompts_path)
    random.shuffle(prompts)
    
    # Optional : define things for keeping tack of step size
    prompts_df = pd.DataFrame(prompts)
    prompts_df = pd.concat([prompts_df, pd.DataFrame(columns=[f'Step{i}' for i in range(args.refining_steps)])], axis=1)
    logging_interval = 1000

    # Dimensions of the latent
    c, w, h = 4, 96, 96 
    
    # Define the output directory
    img_dir = "images-guidance_" + str(args.guidance_level) + "-inf_steps_" + str(args.inference_steps) + "-ref_steps_" + str(args.refining_steps) + "-step_size_" + str(args.step_size) + "-norm_" + str(args.normalize_latents) + "-clip_" + str(args.clip_grads) + "-clip_thresh_" + str(args.clip_threshold) + "-guid_form_" + str(args.guidance_formula) + "-noise_" + str(args.noise_level) + "-bound_" + str(args.boundary_control)
    output_dir = os.path.join(args.output_dir, img_dir)
    
    # Generate images
    for img_idx in tqdm(range(0, args.total_images, args.batch_size), 
                    desc="Generating Images", 
                    total=args.total_images // args.batch_size, 
                    unit='batch'):
        
        # Calculate number of images for this batch
        img_idx_end = min(img_idx+args.batch_size, args.total_images)
        num_prompts = img_idx_end - img_idx
        
        # Selection prompts for this batch
        prompt_batch = prompts[img_idx:img_idx+num_prompts]
        p = [img['caption'] for img in prompt_batch]
        ids = [img['id'] for img in prompt_batch]
        
        # Sample random noise 
        noise = torch.randn(num_prompts, c, w, h, device=device).half()
        
        # Refine noise
        if args.refining_steps > 0:
            with torch.no_grad():
                noise, step_lens = refine_noise(noise, p, device, stable_diffusion, args)
        
        # Generate images with stable diffusion
        with torch.no_grad():
            generated_images = stable_diffusion(
                    p, 
                    latents=noise, 
                    guidance_scale=args.guidance_level, 
                    num_inference_steps=args.inference_steps, 
                    output_type="numpy"
                ).images
        
        # Save generated images to disk
        save_images_from_numpy(generated_images, prompt_batch, output_dir)
        
        # Record length of every step
        if args.refining_steps > 0:
            for step, len_list in zip(range(0, args.refining_steps), step_lens):
                prompts_df.loc[prompts_df["id"].isin(ids), f"Step{step}"] = len_list 
            # Save length of every step
            if img_idx % logging_interval == 0:
                prompts_df.to_csv(os.path.join(output_dir, 'step_sizes.csv'), index=False) 
    

if __name__ == "__main__":

    # Run main
    main()
