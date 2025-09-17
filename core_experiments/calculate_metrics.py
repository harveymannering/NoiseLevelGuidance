import torch
import random
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from utils import load_metric_arguments, set_seed, load_ms_coco_prompts, calculate_clip_score, calculate_inception_score, calculate_fid, calculate_worst_clip_scores

def main():
    
    # Are we using the CPU or the GPU?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load command line arguments
    args = load_metric_arguments()
    
    # Set random seed for reproducible results
    set_seed(0)
    
    # Load and shuffle prompts to be used for generation
    prompts = load_ms_coco_prompts(args.prompts_path)
    random.shuffle(prompts)
    
    # Calculate FID
    print("FID:", calculate_fid(args.fake_image_dir, args.real_image_dir, prompts, device), flush=True)
    
    # Calculate Inception Score (IS)
    print("Inception Score:", calculate_inception_score(args.fake_image_dir, device), flush=True)
    
    # Calculate the CLIP score
    clip_score, df = calculate_clip_score(args.fake_image_dir, prompts, device)
    print("CLIP Score:", clip_score, flush=True)
    
    # Save CLIP Scores
    total_images = 1000
    if args.clip_scores_save_path is not None:
        df.to_csv(args.clip_scores_save_path, index=False)  
        df = df.nsmallest(total_images, 'clip_score')
        prompts = df.to_dict(orient="records")
        clip_score, df = calculate_worst_clip_scores(args.fake_image_dir, prompts, device)
        print(f"CLIP Score (Bottom {total_images} Images):", clip_score, flush=True)
    
    # Calculate CLIP Scores for worst images
    if args.clip_scores_load_path is not None:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(args.clip_scores_load_path) 
        df = df.nsmallest(total_images, 'clip_score')
        prompts = df.to_dict(orient="records")
        clip_score, df = calculate_worst_clip_scores(args.fake_image_dir, prompts, device)
        print(f"CLIP Score (Bottom {total_images} Images):", clip_score, flush=True)



if __name__ == "__main__":
    
    # Run main
    main()
