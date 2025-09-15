# Noise-Level Diffusion Guidance: Well Begun is Half Done


The random Gaussian noise used to start the diffusion process influences the final output, causing variations in image quality and prompt adherence. Existing noise-level optimization approaches generally rely on extra dataset construction, additional networks, or backpropagation-based optimization, limiting their practicality.The repo contains code for Noise Level Guidance (NLG), a simple, efficient, and general noise-level optimization approach that refines initial noise by increasing the likelihood of its alignment with general guidance — requiring no additional training data, auxiliary networks, or backpropagation. Results for Stable Diffusion v2.1 can be seen below.

<img width="1518" height="649" alt="image" src="https://github.com/user-attachments/assets/1c46c381-586c-4c0f-a783-82feffa8d2e9" />
