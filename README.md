# Noise-Level Diffusion Guidance: Well Begun is Half Done

[![arXiv](https://img.shields.io/badge/arXiv-2509.13936-b31b1b.svg)](https://arxiv.org/abs/2509.13936) 

The random Gaussian noise used to start the diffusion process influences the final output, causing variations in image quality and prompt adherence. Existing noise-level optimization approaches generally rely on extra dataset construction, additional networks, or backpropagation-based optimization, limiting their practicality.The repo contains code for Noise Level Guidance (NLG), a simple, efficient, and general noise-level optimization approach that refines initial noise by increasing the likelihood of its alignment with general guidance — requiring no additional training data, auxiliary networks, or backpropagation. An example for Stable Diffusion v2.1 can be seen below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/69d47121-1d64-46f6-8364-12828e0fd1b6" alt="alignment_progression" width="60%">
</p>

## Notebooks

For the most clear demonstration of our method, we recommend looking at our Jupyter notebooks.  We have two notebooks, one for Stable Diffusion v1.5 and one for Stable Diffusion v2.1.  Both contain minimalist implementations of our method with explanations around each step of the algorithm.  Notebooks can be run on Google Colab, or locally if you have an NVIDIA GPU.

## Experiments


Within the `core_experiments` directory you will find code we used to run our Stable Diffusion v2.1 experiments.  To run our method, first download the MS COCO prompts from: https://cocodataset.org/#download.  To use our method can run the following command from within the `core_experiments` folder.  To compare against normal SD2.1 generation, with normal Gaussian noise, change the `--refining_steps` to 0.

```
python generate_images.py --output_dir  ./ \\
  --prompts_path ./MSCOCO/annotations_trainval2014/annotations/captions_val2014.json \\
  --refining_steps 20 \\
  --total_images 4 \\
  --clip_grads \\
  --normalize_latents \\
  --guidance_level 0.0 \\
  --clip_threshold 0.5 \\
  --add_noise \\
  --noise_level 0.001
```

**Dependancies:**

```
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate scipy safetensors
pip install torchmetrics
pip install torch-fidelity
```
