## Implementation

Install diffusion dependencies with:

`pip install -r requirements.txt`

Homework 6 contains two coding problems:

- `gan`: class-conditioned GAN training in the 16D latent space of a pretrained MNIST VAE
- `diffusion`: CLIP-score guided LoRA fine-tuning of Stable Diffusion 1.5

Both problems are designed in the same style as `hw3` and `hw5`:

- students implement only functions whose docstrings begin with `Implement:`
- visible train/val data is provided
- hidden test evaluation is performed from saved checkpoints
- the student release should come from `code/template/`

## gan

### Solution code

`gan/gan_solution.py`

### Data files

`gan/gan_gendata.py` should generate or package:

- `data/train_latents.pt`
- `data/test_latents.pt`
- `data/mnist_vae.pt`

The latent split files should contain:

- `latents`: tensor of shape `(N, 16)`
- `labels`: tensor of shape `(N,)`

`mnist_vae.pt` should contain the pretrained decoder checkpoint used to decode
generated latent points back to MNIST images.

TA/release note:

- `train_latents.pt` and `mnist_vae.pt` are the visible files released to students
- `test_latents.pt` is a hidden autograder file produced by the same `gan_gendata.py` pipeline
- students should not need `test_latents.pt` for `--mode train`
- the autograder should provide `test_latents.pt` when running `python gan_solution.py --mode test`

### Intended training setup

This problem starts from a pretrained unconditional VAE trained by course staff.
Students should:

- load `data/train_latents.pt`
- train a simple class-conditioned generator and discriminator in latent space with one fixed configuration
- save `outputs/gan_model.pt`
- save `outputs/gan_generated_latents.pt`
- generate exactly 10,000 class-conditioned latent samples, 1,000 for each digit class
- decode and save one generated sample per class using the provided unconditional decoder checkpoint

Autograder evaluation should use hidden `data/test_latents.pt` to compare:

- a stratified random sample from `train_latents.pt` against `test_latents.pt`
- the generated 10,000 latent samples against `test_latents.pt`

The grading metric should be the latent-space Fr\'echet distance directly.
Lower is better. Do not add a secondary remapped score.

### Autograder behavior

- unit tests target functions whose docstrings begin with `Implement:`
- running the script in train mode saves `outputs/gan_model.pt`
- running the script also saves `outputs/gan_generated_latents.pt`
- the checkpoint should contain the trained generator plus minimal metadata
- running with `--mode test` loads that checkpoint, regenerates `outputs/gan_generated_latents.pt`, and evaluates against hidden `data/test_latents.pt` when that file is available

## diffusion

### Solution code

`diffusion/diffusion_solution.py`

### Eval code

`diffusion/diffusion_eval.py`

### Data files

`diffusion/diffusion_gendata.py` should generate or package:

- `data/train.jsonl`
- `data/finetune_images/`

The visible training JSONL should contain one row per prompt/image pair with:

- `id`
- `prompt`
- `image_path`

TA/release note:

- `train.jsonl` and `finetune_images/` are the visible files released to students
- `diffusion_gendata.py` may use a visible reward model to filter candidates, but the saved training rows should expose only prompt/image pairs
- the separate 10-prompt evaluation set used by `diffusion_solution.py` is defined directly in the released solution code
- hidden autograder evaluation prompts are not released to students and should be handled only by the autograder-side `diffusion_eval.py`
- TA grading note: on the visible 10-prompt evaluation set, a reasonable base-model score is around `4.0`, while a solid fine-tuned score is around `5.4`; use this as a grading reference rather than as a strict equality test

### Intended training setup

This problem starts from Stable Diffusion 1.5. Students should:

- use `data/train.jsonl` as the visible prompt/image fine-tuning set
- load SD 1.5 from `diffusers`
- freeze the VAE and text encoder, and fine-tune the UNet
- shuffle the visible prompt/image rows each epoch
- train with one fixed default configuration
- print a LaTeX table with the final training loss to stdout
- save 10 base-model generations and 10 fine-tuned generations for the visible evaluation prompts
- save a 4-column comparison image where the left two columns are base generations and the right two columns are fine-tuned generations

### Autograder behavior

- unit tests target functions whose docstrings begin with `Implement:`
- running `python diffusion_solution.py` should train on the visible prompt/image data and save the visible base/fine-tuned evaluation images
- the released solution should not depend on hidden prompts or hidden data files
- autograder-side evaluation should be handled by `python diffusion_eval.py ...` on hidden prompt generations or hidden saved manifests/bundles
- students should not need hidden prompt files locally
- TA/autograder note: visible reporting is worth 8 points for showing the required images and 6 points for score quality; a base score around `4.0` and fine-tuned score around `5.4` is an appropriate target range for full credit

### Eval usage

By default, the visible evaluator scores the saved fine-tuned image directory:

`python diffusion_eval.py`

This defaults to:

- `outputs/finetuned_eval_images`

To score the base-model images instead:

`python diffusion_eval.py --image_dir outputs/base_eval_images`

Optional alternate inputs are also supported:

- `--image_dir ...`
- `--manifest_path ...`
- `--bundle_path ...`
