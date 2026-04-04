## Implementation

Install dependencies with:

`pip install -r requirements.txt`

Homework 6 has two coding parts:

- `gan`: class-conditioned GAN training in a pretrained MNIST VAE latent space
- `diffusion`: LoRA fine-tuning for Stable Diffusion 1.5

As in earlier homeworks:

- students only implement functions marked with `Implement:`
- visible data is provided to students
- hidden evaluation is done from saved outputs/artifacts
- the student-facing release should come from `code/template/`

## GAN

Main file:

- `gan/gan_solution.py`

Data/setup:

- generated from `gan/gan_gendata.py`
- uses visible training latents and a pretrained MNIST decoder
- hidden test latents are reserved for autograding

Expected student workflow:

- train a class-conditioned GAN in latent space
- generate class-conditioned latent samples
- decode representative outputs back to MNIST images

TA note:

- grade primarily from generated latent outputs
- hidden evaluation compares generated latents against held-out latent data

## Diffusion

Student file:

- `diffusion/diffusion_solution.py`

Autograder file:

- `diffusion/diffusion_eval.py`

Data/setup:

- generated from `diffusion/diffusion_gendata.py`
- visible training data is prompt/image pairs
- hidden prompts or hidden evaluation assets are reserved for autograding

Expected student workflow:

- fine-tune Stable Diffusion 1.5 on the visible dataset
- save base-model and fine-tuned generations for the visible prompts
- report the required comparison outputs

TA note:

- visible outputs include a written-report comparison figure plus submitted fine-tuned eval images
- `diffusion/diffusion_eval.py` should remain TA/autograder-side, not student-facing
- autograder-side scoring should run on submitted fine-tuned eval images
