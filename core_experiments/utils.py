import argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import json
import numpy as np
import random
import os 
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T
from pathlib import Path
from typing import Tuple, Union, List
from torch import Tensor
from torchmetrics.functional.multimodal.clip_score import _clip_score_update
import pandas as pd

# Object to calculate CLIP Score which also track individual CLIP Scores of each text/image pair
class CLIPScoreWithIndividual(CLIPScore):
    def __init__(self, model_name_or_path: str = "openai/clip-vit-base-patch16", **kwargs):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.individual_scores = []

    def update_and_get(self, source: Union[Tensor, List[Tensor], List[str], str], 
                target: Union[Tensor, List[Tensor], List[str], str]) -> Tensor:
        score, n_samples = _clip_score_update(source, target, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples
        self.individual_scores.extend(score.tolist())
        return score

    def compute(self) -> Tensor:
        return super().compute()

    def reset(self) -> None:
        super().reset()
        self.individual_scores = []

def calculate_clip_score(image_dir, prompts, device):

    # Create transform pipeline
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize edge to 224
        T.ToTensor(),  # Convert to tensor and scale to [0, 1]
        T.Lambda(lambda x: (x * 255).byte())  # Convert to uint8 (0-255)
    ])
    
    # Get all image files and formate prompts
    files = sorted(Path(image_dir).glob('*.png'))
    prompts_df = pd.DataFrame(prompts)
    prompts_df['clip_score'] = float('nan')
    
    # Define object for calculate CLIP score
    metric = CLIPScoreWithIndividual(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    # iterate over all images
    for f_idx in range(len(files)):
        
        # Get prompt used to generate the image 
        id = int(files[f_idx].name.split('.')[0])
        p = prompts_df[prompts_df['id'] == id]['caption'].iloc[0]
        
        # Load image using PIL
        image = Image.open(files[f_idx])
        tensor = transform(image).to(device)
        
        # Accumulate CLIP score
        with torch.no_grad():
            score = metric.update_and_get(tensor, p)
            prompts_df.loc[prompts_df['id'] == id, 'clip_score'] = score.item()
        
    # Calculate CLIP score for a large number of images
    return metric.compute(), prompts_df.head(len(files))

def calculate_worst_clip_scores(image_dir, prompts, device):

    # Create transform pipeline
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize edge to 224
        T.ToTensor(),  # Convert to tensor and scale to [0, 1]
        T.Lambda(lambda x: (x * 255).byte())  # Convert to uint8 (0-255)
    ])
    
    # Get all image files and formate prompts
    prompts_df = pd.DataFrame(prompts)
    
    # Define object for calculate CLIP score
    metric = CLIPScoreWithIndividual(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    # iterate over all rows
    for index, row in prompts_df.iterrows():
        
        # Get prompt used to generate the image 
        id = row["id"]
        p = row["caption"]
        
        # Load image using PIL
        image = Image.open(os.path.join(image_dir, str(id) + ".png"))
        tensor = transform(image).to(device)
        
        # Accumulate CLIP score
        with torch.no_grad():
            score = metric.update_and_get(tensor, p)
            prompts_df.loc[prompts_df['id'] == id, 'clip_score'] = score.item()
    
    # Calculate CLIP score for a large number of images
    return metric.compute(), prompts_df

def calculate_inception_score(image_dir, device):

    # Create transform pipeline
    transform = T.Compose([
        T.Resize((299, 299)),  # Resize shortest edge to 224
        T.ToTensor(),  # Convert to tensor and scale to [0, 1]
        T.Lambda(lambda x: (x * 255).byte())  # Convert to uint8 (0-255)
    ])
    
    # Get all image files and formate prompts
    files = sorted(Path(image_dir).glob('*.png'))
    
    # Define object for calculate IS
    inception = InceptionScore().to(device)

    # iterate over all images
    for f_idx in range(len(files)):
        
        # Load image using PIL
        image = Image.open(files[f_idx])
        tensor = transform(image).to(device)
        
        # Accumulate IS
        inception.update(tensor[None])
    
    # Calculate inception score for a large number of images
    return inception.compute()

def calculate_fid(fake_image_dir, real_image_dir, prompts, device):

    # Create transform pipeline
    transform = T.Compose([
        T.Resize((299, 299)),  # Resize shortest edge to 224
        T.ToTensor(),  # Convert to tensor and scale to [0, 1]
        T.Lambda(lambda x: (x * 255).byte())  # Convert to uint8 (0-255)
    ])
    
    # Get all image files and formate prompts
    fake_files = sorted(Path(fake_image_dir).glob('*.png'))
    prompts_lookup = {item['id']: item for item in prompts}
    
    # Define object for calculate FID score
    fid = FrechetInceptionDistance().to(device)
    
    # Load paths for real images
    real_files = sorted(Path(real_image_dir).glob('*.jpg'))
    random.shuffle(real_files)

    # iterate over all images
    for f_idx in range(len(fake_files)):
        
        # Load fake image using PIL
        fake_image = Image.open(fake_files[f_idx])
        fake_tensor = transform(fake_image).to(device)
        
        # Load real image using PIL
        real_image = Image.open(real_files[f_idx]).convert('RGB')
        real_tensor = transform(real_image).to(device)
        
        # Accumulate IS
        fid.update(real_tensor[None], real=True)
        fid.update(fake_tensor[None], real=False)
    
    # Calculate inception score for a large number of images
    return fid.compute()

def save_images_from_numpy(image_array, prompt_batch, output_folder):
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through images and save
    for i, img in enumerate(image_array):
        # Ensure image is in the right format for PIL
        scaled_img = (img * 255)
        img = np.clip(scaled_img, 0, 255).astype(np.uint8)
        
        # Create PIL Image
        pil_img = Image.fromarray(img)
        
        # Generate filename
        filename = os.path.join(output_folder, f"{prompt_batch[i]['id']}.png")
        
        # Save image
        pil_img.save(filename)
    
    print(f"Saved {len(image_array)} images to {output_folder}")

def load_generation_arguments():
    
    # Define the command line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Output directory where images will be saved.")
    parser.add_argument("--prompts_path", required=True, help="Path to the json file where prompts are stored.")
    parser.add_argument("--stable_diffusion_path", default="stabilityai/stable-diffusion-2-1", help="Path to the stabilityai/stable-diffusion-2-1 latent diffusion model.")
    parser.add_argument("--total_images", default=30000, type=int, help="How many images are we generating?")
    parser.add_argument("--inference_steps", default=20, type=int, help="Total number of steps used to generate images.")
    parser.add_argument("--batch_size", default=8, type=int, help="Number of images we generate in parallel.")
    parser.add_argument("--refining_steps", default=0, type=int, help="Total number of steps used to refine noise.")
    parser.add_argument("--step_size", default=1, type=float, help="Step size used during noise refining.")
    parser.add_argument("--guidance_level", default=7.5, type=float, help="What guidance scale are we using?")
    parser.add_argument("--clip_threshold", default=1.0, type=float, help="What the max length of an edit direction (only if clip_grads=True)")
    parser.add_argument("--noise_level", default=0.0, type=float, help="How much additional noise should we add?")
    parser.add_argument("--clip_grads", action='store_true', default=False, help="Set edit direct to have a maximum length.")
    parser.add_argument("--normalize_latents", action='store_true', default=False, help="What guidance scale are we using?")
    parser.add_argument("--guidance_formula", action='store_true', default=False, help="Use the original guidance formula")
    parser.add_argument("--add_noise", action='store_true', default=False, help="Add a small amount of noise to the latent at every step")
    parser.add_argument("--boundary_control", action='store_true', default=False, help="Clip the latent to 3 standard deviations at each step")
    
    # Load arguments in to variables
    args = parser.parse_args()
    return args

def load_metric_arguments():
    
    # Define the command line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_image_dir", required=True, help="Directory where real images have been saved.")
    parser.add_argument("--fake_image_dir", required=True, help="Directory where fake images have been saved.")
    parser.add_argument("--prompts_path", required=True, help="Path to the json file where prompts are stored.")
    parser.add_argument("--clip_scores_save_path", default=None, type=str, help="Where to save the CLIP scores")
    parser.add_argument("--clip_scores_load_path", default=None, type=str, help="Where to load previously calculated CLIP scores")


    # Load arguments in to variables
    args = parser.parse_args()
    return args

def load_stable_diffusion(path, device):
    
    # Load stable diffusion, either from the internet or from a local path
    pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, local_files_only=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

def load_ms_coco_prompts(json_path):
    
    # Load MS COCO 2014 prompts which is stored in a json file (source: https://cocodataset.org/#download)
    with open(json_path) as f:
        meta_data = json.load(f)
    return meta_data['annotations']

def set_seed(seed):
    
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def refine_noise(latents, prompts, device, pipe, args):
    
    # Get latent dimensions
    b, c, w, h = latents.shape
    
    # Duplicate latents for CFG
    t = torch.tensor(999, device=device)
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompts,device,1,True)
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    #  Calculate total dimension of the image
    dimensions = len(torch.flatten(latents[0]))
    
    # Define record for step lengths
    step_magnitudes = []

    for it in range(args.refining_steps):
        # Copy noise for conditional/unconditorch.flatten()tion generation
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input.half(),
            t,
            encoder_hidden_states=prompt_embeds.half(),
            return_dict=False,
        )[0]

        # Convert v-prediction to ε-prediction
        sigma = pipe.scheduler.sigmas[0]
        alpha_t, sigma_t = pipe.scheduler._sigma_to_alpha_sigma_t(sigma)
        epsilon = alpha_t * noise_pred + sigma_t * latent_model_input
        
        # Find the direction
        if args.guidance_formula == True:
            gamma = 3.0
            dir = (gamma * epsilon[b:] + (1-gamma) * epsilon[:b]).detach()
        else:
            dir = (epsilon[b:] - epsilon[:b]).detach()
        
        # Record the lengths of the steps
        step_magnitudes.append(torch.norm(dir.view(-1, c * w * h), dim=1).tolist())

        # Gradient norm clipping of direction
        if args.clip_grads == True:
            clip_mag = args.clip_threshold
            dir_norm = torch.norm(dir.view(-1, c * w * h), dim=1)
            mask = dir_norm > clip_mag
            dir = torch.where(
                mask[:, None, None, None], 
                clip_mag * (dir / dir_norm[:, None, None, None]), 
                dir
            ).view(-1, c, w, h)
        
        # Push the noise in a particular direction
        latents -= dir * args.step_size
        
        # Add a small amount of noise at every step (following arxiv:2303.13703)
        if args.add_noise == True:
            np_state = np.random.get_state()
            latents += torch.tensor(np.random.randn(b, c, w, h), device=device).half() * np.sqrt(args.noise_level) #additional_noise[it:it+1].to(device) * np.sqrt(10e-4)
            np.random.set_state(np_state)
        
        # Boundary control following NoiseDiffusion (arxiv:2403.08840)
        if args.boundary_control == True:
            latents = torch.clamp(latents, -3, 3)
        
        # Normalize latent to be the expected magnitude
        if args.normalize_latents == True:
            latent_norms = torch.norm(latents.view(-1, c * w * h), dim=1)[:,None,None,None]
            latents = (latents / latent_norms) * np.sqrt(dimensions)

    return latents, step_magnitudes
