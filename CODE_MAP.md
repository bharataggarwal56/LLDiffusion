# LLDiffusion code map

Read from source only; nothing was run. `data/` does not exist in this checkout and there are no checkpoints. Line references are approximate.

## 1. Big picture

LLDiffusion is a conditional DDPM/DDIM for low-light enhancement. The network predicts the noise added to the normal-light image (`hr`). The low-light image (`lr`) is the condition.

There are three conditioning paths:
1. `lr` is concatenated to the noisy `hr` at the input.
2. A **colour map** computed from `lr` is concatenated at the input.
3. A **degradation encoder** looks at `lr` and injects multi-scale features into the decoder of the diffusion UNet (`fuse_mode: cattn`).

A second UNet (`deg_unet`) is used only in training. It re-degrades `hr` back into `lr` using the same degradation features. This forces the encoder to capture the low-light degradation.

File flow: `train.py` → `datasets/lol.py` (`LOL`) → `models/ddm.py` (`DenoisingDiffusion.train`) → `models/losses.py` (`compute_loss` → `deg_loss`) → `models/modules/degradation/__init__.py` (`DegrationUNet`) → `models/modules/unet/*` (`DiffusionUNet`).

## 2. `train.py` (65 lines)

- Parses `--config` (required), `--resume`, `--sampling_timesteps` (default **25**), `--image_folder` and `--seed` (default 0).
- Loads the YAML into nested `argparse.Namespace` objects with `dict2namespace`, so config values are read as `config.data.root` and so on.
- Sets `config.device`, seeds torch and numpy, and enables `cudnn.benchmark`.
- Builds the dataset with `datasets.__dict__[config.data.dataset](config)`, which is the `LOL` class.
- Builds `DenoisingDiffusion(args, config)` and calls `.train(DATASET)`. It does nothing else.

`DenoisingDiffusion` (`models/ddm.py`) holds the training logic:
- **Setup.**
  - It builds the model through `get_model`, which wraps it in `nn.DataParallel`.
  - It builds an Adam optimizer from `utils/optimize.py` and an `EMAHelper`.
  - It builds a linear beta schedule, 1e-4 to 0.02 over 1000 steps.
  - It builds a `MultiStepLR` scheduler, a TensorBoard writer and a loguru log. All of these live under `data.workspace`.
- **Loop**, per epoch and per batch:
  1. `lows`/`highs` are in [0,1] and go through `data_transform` to [-1,1]. They are concatenated into `x` with 6 channels.
  2. The noise is `e = randn_like(lows)`.
  3. Timesteps use **antithetic sampling**: draw `n//2+1` values of `t`, then append `T-1-t` and truncate to `n`.
  4. `compute_loss(...)` returns a dict of losses. They are weighted by `config.params.loss_weights`, summed, and sent through backward, optimizer step and EMA update.
  5. After the epoch, the scheduler steps and the mean losses are logged. Then `validate()` runs.
- **`validate`.**
  - It only runs when `(epoch+1) % validation_freq == 0`.
  - It restores each test image with `restore_single` (full image, `r=None`) and computes PSNR/SSIM.
  - It saves `best-<epoch>-<psnr>.pth` and `last-<epoch>-<psnr>.pth` in `<workspace>/ckpts`, deleting the previous one of each kind.
- Checkpoints store the epoch, step, state_dict, optimizer, EMA shadow, args and config.

## 3. The model (`models/`)

`models/__init__.py` re-exports `ddm`. `models/modules/__init__.py:get_model` dispatches on `config.model.name`. For LOL that is `DegUNet`, which returns `DegrationUNet`.

For DegUNet no forward-pre-hook is registered. The hook that uses `utils/maps.py` is only for the other model names, or when `data.est` is set. The colour map for DegUNet is computed inside the model itself.

### `DegrationUNet` (`models/modules/degradation/__init__.py`)

It has three parts:

**`encoder` = `DegEncoder`** (`degradation/encoder.py`).
- `ConvEncoder` is a spectral-norm/instance-norm conv stack with 5 stride-2 convs, so it downsamples by 32. Then come 3 residual blocks and 3 output convs. The input is `lr` and the output is a 512-channel latent at H/32.
- `PriorUpsample` turns that latent into a list of feature maps. Each step is resize (2×) → reflect-pad → spectral-norm conv → LeakyReLU.
- With LOL's `num_deg_fusion: 3`, only the first 3 stages are kept:
  - channels are `[512, 512, 256]`
  - scales are `[2, 2, 2]`
  - so for a 128 px patch the features are at 8², 16² and 32².

**`diff_unet` = `DiffusionUNet`** (`models/modules/unet/`). This is the denoiser, with time embedding.
- It takes 9 channels: noisy hr (3) + lr (3) + colour map (3). The `in_channels: 9` in the config matches this.
- `ch=64`, `ch_mult=[1,1,2,2,4,4]` (6 levels, total downsampling 32, so H and W must be multiples of 32), 2 res blocks per level and no dropout.
- Self-attention is applied at "scale 8", which is the level at H/4, plus the mid block.
- The up path uses `skip_mode: add` and the timestep embedding has `4*ch` channels.
- The degradation features are injected only in the **up path**, at up-levels 1, 2 and 3. At each one, `Fusion` (`unet/fusion.py`) concatenates the feature with the running activation and passes the result through `ChannelAttention`. This is `cattn`.
- The down path, the mid block and the final layer get no injection.

**`deg_unet` = `DiffusionUNet`** with no time embedding.
- It takes `hr` (3 channels) with `ch = deg_ch = 32`, and the same injected features.
- It outputs `pred_lr`, a reconstruction of the low-light image.
- It is used only when `self.training and with_deg`. At eval time it is skipped.

`forward(x, t, return_latents=False)`:
- In training `x = (noisy_hr, lr, hr)`. At eval it is `(noisy_hr, lr)`.
- The features come from `encoder(lr)`. They are padded into a 12-slot list: `[None]*6 + features + [None]*3`. The `DiffusionUNet` unpacks it into down, mid, up and final injections.
- Training returns `(pred_noise, pred_lr)`. Eval returns `pred_noise`.

`DegrationUNet.encoder` is a different thing from `models/estimator`, which is only used when `data.est` is set.

**Encoder width quirk.** In `DegrationUNet.__init__`, `encoder_config` is rebuilt using `getattr(config.params, "ch", 16)`. It reads `params.ch`, not `params.encoder.ch`. The YAML sets `encoder: {ch: 32, encoder_attn: True, upsample_attn: True}`, but the code never sees those values. The encoder is therefore built with `ch=16` and both attention flags `False`. This looks like a bug, but do not "fix" it blindly. `PriorUpsample` hardcodes 512 as its input width, and `ConvEncoder` outputs `ch*32`, which is 512 only for `ch=16`. Setting `ch=32` would break the shapes.

Other model files (`dit.py`, `sr3.py`, `restore_former/`, `resformer/`, `twin_unet/`, `estimator/`) are alternative baselines and are not used by the LOL config.

## 4. How the colour map is computed and fed in

`compute_color_map` is in `models/modules/degradation/__init__.py`:

```python
x = (x + 1) / 2                                   # [-1,1] -> [0,1]
color_map = x / (x.sum(dim=1, keepdims=True) + 1e-4)   # per-pixel chromaticity r/(r+g+b)
color_map = color_map.clip(0, 1)
color_map = color_map * 2 - 1                     # back to [-1,1]
```

- It is **per-pixel RGB chromaticity**, meaning the colour with brightness divided out. It is computed from `lr`, not from the noisy image or the ground truth.
- It is recomputed on the fly on every forward pass, so the dataset knows nothing about it.
- It is enabled by `params.color`, which defaults to True.
- It is concatenated in `DegrationUNet.forward` as `torch.cat([noisy_hr, lr, color_map], dim=1)`. That makes 9 channels, which `diff_unet` takes as input.

`utils/maps.py` has a separate `compute_maps`/`compute_conditional_inputs`, which also builds a histogram-equalised image and a gradient-based noise map. It is not used by DegUNet because `config.data.maps` is not set.

## 5. Loss function (`models/losses.py:deg_loss`)

`compute_loss` dispatches on the model name. For `DegUNet` it calls `deg_loss`.

1. Forward diffusion: `noisy_hr = sqrt(ᾱ_t)·hr + sqrt(1-ᾱ_t)·e`.
2. The model is called as `model((noisy_hr, lr, hr), t)` and returns `pred_noise, pred_lr`.
3. Losses:
   - `noise_loss = MSE(pred_noise, e)`.
   - `image_loss = L1(inverse_data_transform(pred_lr), inverse_data_transform(lr))`. This is the degradation-reconstruction loss, i.e. how well `hr` plus the encoder features reproduce `lr`.
   - `perceptual_loss` exists but is commented out in the LOL config. It would be an L1 between encoder feature maps of `lr` and `pred_lr`.
4. Weighted total from the config: `1.0 * noise_loss + 0.1 * image_loss`.
5. **Encoder freezing.** Once `epoch >= fix_epoch` (40000), the encoder params get `requires_grad=False` and `image_loss` is dropped. Only the noise loss remains. With `n_epochs: 60000`, the last 20000 epochs train with the noise loss alone.

## 6. Sampling at test time

Code is in `ddm.py` (`diffusive_sample`, `sample_image`, `restore`, `restore_single`) and `test.py`.

- It is **DDIM with eta=0**, so it is deterministic apart from the initial noise.
- The step count comes from `--sampling_timesteps` (default 25 in both `train.py` and `test.py -t`). `skip = 1000 // 25 = 40`, and `seq = range(0, 1000, 40)`. That gives 25 steps, from t=960 down to t=0, then to ᾱ=1.
- Each step:
  1. Predict `et = model((xt, x_cond), t)`.
  2. Compute `x0_t = (xt - sqrt(1-ᾱ_t)·et)/sqrt(ᾱ_t)`. There is no clipping of x0.
  3. Compute `xt_next = sqrt(ᾱ_next)·x0_t + sqrt(1-ᾱ_next)·et`.
- The start is `x = randn` with the same shape as the conditioning image. The result is mapped back with `inverse_data_transform` (clamped to [0,1]).
- There are two restore modes:
  - **Whole image** (`r=None`). This is used by validation during training. The image is reflect-padded to a multiple of 32 (`model.scale`) and sampled in one pass. The output is cropped back.
  - **Overlapping patches** (`test.py`, default `--grid_r 16`, patch = `data.image_size` = 128, patch batch `-b 512`). Patches are cropped on a stride-16 grid, the noise predictions are summed and divided by the per-pixel coverage count. Use `--no-patch` for whole-image mode.
- `test.py` loads the checkpoint with `ema=False`, so it uses the raw weights. It writes PNGs to `--dir` and averages PSNR/SSIM (`utils/metrics.py`).

## 7. LOL config (`configs/deg-unet-lol.yml`)

| Setting | Value |
|---|---|
| Patch size | **128** (`data.image_size`) |
| Patches per image | `training.patch_n: 4` |
| Batch size | **8** images, which is 32 patches per step after `collate_fn` flattens |
| Epochs | **60000** (`n_epochs`). `n_iters: 2000000` is not read anywhere. |
| Sampling steps | **25**. This is the CLI default, not set in the YAML. The model was trained with `num_diffusion_timesteps: 1000`. |
| `sampling.batch_size` | 32. It is the validation/test loader batch size. |
| Validation | every 500 epochs, on the test split |
| Optimizer | Adam, lr 1e-4, no weight decay |
| LR schedule | MultiStepLR, milestones [100, 1000, 5000], gamma 0.5, stepped per epoch |
| Beta schedule | linear 1e-4 → 0.02, 1000 steps |
| Loss weights | noise 1, image 0.1 |
| `fix_epoch` | 40000 |
| `fuse_mode` / `skip_mode` | `cattn` / `add` |
| `num_deg_fusion`, `deg_ch` | 3, 32 |
| UNet | `ch: 64`, `ch_mult: [1,1,2,2,4,4]`, 2 res blocks, `attn_resolutions: [8]`, in 9 and out 3 channels |
| Data | `root: data/sets/LOL`, `workspace: data/logs/lol`, 8 workers |

One "epoch" is one pass over the training images. For the standard 485-image LOL train set that is about 61 steps per epoch.

## 8. Paths the LOL dataset loader expects (`datasets/lol.py`)

With `data.root: data/sets/LOL` (relative to where you run the script), the layout is:

```
data/sets/LOL/
├── train/
│   ├── low/   <id>.png
│   └── high/  <id>.png
└── test/
    ├── low/   <id>.png
    └── high/  <id>.png
```

- `train_loader` reads `<root>/train`. `val_loader` and `test_loader` both read `<root>/test`.
- The code lists `*/low/*.png` (PNG only, sorted). The ground truth is the file with the **same filename** in `high/`.
- **Filenames must have an integer stem** (`1.png`, `22.png`). `get_image` does `int(stem)` and uses it as the label, so names like `low00001.png` crash.
- The original LOL download uses `our485/` for train and `eval15/` for test. You need to rename or move those into `train/` and `test/` yourself.
- The `filelist` mode (a text file of `input` → `gt` paths) exists in `LOLDataset` but is not used by the `LOL` class.
- Training takes 4 random 128×128 crops per image. Augmentation is commented out. Validation and test use whole images with no crop.

## 9. Things to watch

- **Case mismatch.** The README says to extract into `data/sets/lol`, but the configs say `data/sets/LOL`. Linux is case-sensitive, so make them match.
- **README test command.** It references `configs/deg-unet.yml`, which does not exist. It should be `configs/deg-unet-lol.yml`. The v2 command also uses `deg-unet-lol-v2-syn.yml`, while the file is `deg-unet-lol-v2-sys.yml`.
- **EMA is never used.** `ema_rate: 0.999` is not read, because `EMAHelper()` uses its default 0.9999. The shadow weights are saved but never applied at validation or in `test.py`.
- **LR scheduler resume.** The code stores the scheduler as `self.lr_schduler` (typo), but `load` looks for `self.lr_scheduler`. After `--resume`, the optimizer state is restored and the scheduler state is not.
- **Possible NaN at image borders in patch mode (reasoned, not run).** The patch grid is `range(0, W-128+1, 16)`. For 600 px width the last patch starts at 464 and ends at 592, so the last 8 columns are covered by no patch. The division by the coverage count would then give NaN or inf there. Whole-image mode (`--no-patch`) avoids this. The README results may have been produced differently.
- **Checkpoint saving inside the batch loop.** In `validate`, the best/last checkpoint code runs inside the data-loader loop. It works on the 15-image LOL test set, because the batch size is 32 and there is one batch.
- **Repo hygiene.** `models/modules/unet/__pycache__/*.pyc` is committed, and `venv/` is untracked.
