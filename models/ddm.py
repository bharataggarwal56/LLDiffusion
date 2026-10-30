import math
import os
import time

import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import pad
from torchvision.transforms.functional import crop
from torchvision.utils import save_image

from utils.config import DDIM_STEPS, resolve_resume
from utils.metrics import calculate_psnr, calculate_ssim
from utils.optimize import get_optimizer
from utils.transforms import data_transform, inverse_data_transform

from .ema import EMAHelper
from .losses import compute_loss
from .modules import get_model


def compute_betas(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                 num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end,
                            num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps,
                                  1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def patch_e(x, x_cond, model, corners, p_size, batch_size, t, xt):
    et_output = torch.zeros_like(x_cond, device=x.device)
    xt_patches = torch.cat([
        crop(xt, hi, wi, p_size, p_size)
        for (hi, wi) in corners
    ], dim=0)
    x_cond_patches = torch.cat([
        crop(x_cond, hi, wi, p_size, p_size)
        for (hi, wi) in corners
    ], dim=0)

    xt_batches = xt_patches.split(batch_size)
    x_cond_batches = x_cond_patches.split(batch_size)
    # TODO: This is a hack to make it work with multiple GPUs
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    index = 0
    for xt_batch, x_cond_batch in zip(xt_batches, x_cond_batches):
        outputs = model((xt_batch, x_cond_batch), t)
        for output in outputs:
            hi, wi = corners[index]
            et_output[0, :, hi:hi + p_size, wi:wi + p_size] += output
            index += 1
    assert index == len(corners)
    return et_output


def pad_to_multiple(x, m):
    """Reflect-pad the bottom/right of x so that H and W are multiples of m."""
    ph, pw = (-x.shape[-2]) % m, (-x.shape[-1]) % m
    while ph > 0 or pw > 0:
        h, w = x.shape[-2:]
        assert h > 1 and w > 1, "cannot reflect-pad a 1-pixel image"
        a, b = min(ph, h - 1), min(pw, w - 1)
        x = pad(x, (0, b, 0, a), mode="reflect")
        ph, pw = ph - a, pw - b
    return x


def overlapping_grid_indices(x_cond, psize, r=None):
    h, w = x_cond.shape[-2:]
    r = 16 if r is None else r
    h_list = [i for i in range(0, h - psize + 1, r)]
    w_list = [i for i in range(0, w - psize + 1, r)]
    return h_list, w_list


@torch.no_grad()
def diffusive_sample(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, batch_size=8):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    if corners is not None:
        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
        assert (x_grid_mask > 0).all(), "patch grid leaves part of the image uncovered"

    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to(x.device)

        if corners is not None:
            et_output = patch_e(x, x_cond, model, corners,
                                p_size, batch_size, t, xt)
            et = torch.div(et_output, x_grid_mask)
        else:
            et = model((xt, x_cond), t)

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_preds.append(x0_t)

        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        xs.append(xt_next)

    xs = [_.cpu() for _ in xs]
    x0_preds = [_.cpu() for _ in x0_preds]
    return xs, x0_preds


class DenoisingDiffusion:
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = get_model(config)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = get_optimizer(self.config, self.model.parameters())
        self.epoch, self.step = 0, 0
        self.lr_scheduler = None

        self.betas = compute_betas(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        ).to(self.device)

        betas = self.betas
        self.num_timesteps = betas.shape[0]

        os.makedirs(self.workspace, exist_ok=True)
        log_dir = os.path.join(self.workspace, "tensorboard")
        self.writer = SummaryWriter(log_dir)

        level = "INFO"
        log_filename = os.path.join(self.workspace, "output.log")
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        logger.add(log_filename, level=level, format=file_format)
        self.logger = logger

        lr_scheduler = getattr(config, "lr_scheduler", None)
        if lr_scheduler is not None:
            if lr_scheduler.name == "MultiStepLR":
                from torch.optim.lr_scheduler import MultiStepLR
                kwargs = vars(lr_scheduler.kwargs)
                self.lr_scheduler = MultiStepLR(self.optimizer, **kwargs)
            else:
                msg = f"lr_scheduler {lr_scheduler.name} is not implemented"
                raise NotImplementedError(msg)
        self.best_score = -1.0
        self.best_epoch = 0

    def load(self, load_path, ema=False):
        # our own checkpoints pickle the argparse config, so weights_only must be off
        checkpoint = torch.load(load_path, map_location=self.device,
                                weights_only=False)

        if "epoch" not in checkpoint.keys():
            self.logger.info(f"loaded: {load_path} (pretrained)")
            self.model.load_state_dict(checkpoint, strict=False)
            return

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.ema_helper.load_state_dict(checkpoint["ema_helper"])

        if self.lr_scheduler is not None:
            state = checkpoint.get("lr_scheduler")
            if state is not None:
                self.lr_scheduler.load_state_dict(state)

        if ema:
            self.ema_helper.ema(self.model)
        self.logger.info(
            f"loaded {load_path} (epoch {self.epoch}, step {self.step})")

    def _stop_reason(self, max_iters, time_budget_hours, start_time):
        if max_iters is not None and self.step >= max_iters:
            return f"max_iters ({max_iters}) reached"
        if (time_budget_hours is not None
                and time.time() - start_time >= time_budget_hours * 3600):
            return f"time budget ({time_budget_hours}h) reached"
        return None

    def train(self, dataset):
        self.logger.info("start training")
        train_loader = dataset.train_loader()
        val_loader = dataset.val_loader()

        resume = resolve_resume(self.args.resume, self.config)
        if resume != "":
            self.load(resume)
        elif self.args.resume == "auto":
            self.logger.info("--resume auto: no checkpoint found, starting fresh")

        tc = self.config.training
        max_iters = getattr(tc, "max_iters", None)
        time_budget_hours = getattr(tc, "time_budget_hours", None)
        ckpt_every = getattr(tc, "ckpt_every", None)
        log_every = getattr(tc, "log_every", 50)
        freq_weight = getattr(tc, "freq_loss_weight", 0)

        params = getattr(self.config, "params", {})
        loss_weights = getattr(params, "loss_weights", None)
        loss_weights = None if loss_weights is None else dict(vars(loss_weights))

        def weight_of(k):
            if k.startswith("freq_"):
                return freq_weight
            return 1.0 if loss_weights is None else loss_weights[k]

        self.logger.info(
            f"start training from epoch {self.epoch}, step {self.step} "
            f"(max_iters={max_iters}, time_budget_hours={time_budget_hours})")
        start_time = last_log_time = time.time()
        start_epoch = self.epoch
        stop = self._stop_reason(max_iters, time_budget_hours, start_time)
        for epoch in range(start_epoch, tc.n_epochs):
            if stop:
                break
            self.epoch = epoch
            total_loss = 0.0
            loss_dict = {}
            n_steps = 0
            for x, y in train_loader:
                lows, highs = x
                lows, highs = lows.to(self.device), highs.to(self.device)
                lows, highs = data_transform(lows), data_transform(highs)
                x = torch.cat([lows, highs], dim=1)

                e = torch.randn_like(lows)
                b = self.betas

                # antithetic sampling
                n = x.size(0)
                t = torch.randint(self.num_timesteps,
                                  (n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = compute_loss(self, x, t, e, b)
                step_terms = {}
                if isinstance(loss, dict):
                    for k, v in loss.items():
                        step_terms[k] = v.item()
                        loss_dict[k] = loss_dict.get(k, 0.0) + v.item()
                    loss = sum([loss[k] * weight_of(k) for k in loss])

                assert torch.isfinite(loss), f"non-finite loss: {step_terms}"
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)

                n_steps += 1
                self.step += 1
                for k, v in step_terms.items():
                    self.writer.add_scalar(f"iter/{k}", v, self.step)
                if log_every and self.step % log_every == 0:
                    now = time.time()
                    its = log_every / max(now - last_log_time, 1e-9)
                    last_log_time = now
                    if torch.cuda.is_available():
                        mem = f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                    else:
                        mem = "n/a (no cuda)"
                    self.logger.info(
                        f"step: {self.step}, loss: {loss.item():.4f}, " + ", ".join(
                            f"{k}: {v:.4f}" for k, v in step_terms.items())
                        + f", it/s: {its:.2f}, peak_mem: {mem}")
                    self.writer.add_scalar("perf/it_per_sec", its, self.step)
                if ckpt_every and self.step % ckpt_every == 0:
                    self.save_resumable()
                stop = self._stop_reason(max_iters, time_budget_hours, start_time)
                if stop:
                    break

            if n_steps > 0:
                loss = total_loss / n_steps
                loss_dict = {k: v / n_steps for k, v in loss_dict.items()}
                self.logger.info(f"epoch: {epoch}, step: {self.step}, loss: {loss:.4f}, " + ", ".join([
                    f"{k}: {v:.4f}" for k, v in loss_dict.items()
                ]))
                self.writer.add_scalar("train/total_loss", loss, epoch)
                self.writer.add_scalars("train/loss", loss_dict, epoch)

            if stop:
                break
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.validate(val_loader)

        # an interrupted epoch is resumed from its start; a finished one from the next
        self.save("final.pth", epoch=self.epoch if stop else self.epoch + 1)
        self.logger.info(f"training finished at step {self.step}: "
                         f"{stop or 'n_epochs reached'}; saved final.pth")

    @torch.no_grad()
    def validate(self, data_loader, r=None, save_images=False):
        epoch = self.epoch
        if (epoch + 1) % self.config.training.validation_freq != 0:
            validation_epochs = getattr(
                self.config.training, "validation_epochs", [])
            if (epoch + 1) not in validation_epochs:
                return

        self.logger.info(f"start validation at epoch {epoch}")
        training = self.model.training
        self.model.eval()

        psnrs, ssims = [], []
        for x, labels in data_loader:
            lows, highs = x
            lows, highs = lows.to(self.device), highs.to(self.device)

            for low, high, label in zip(lows, highs, labels):
                pred = self.restore_single(low, r=r)
                h, w = high.shape[-2:]
                pred = pred[:, :h, :w]
                psnr = calculate_psnr(pred, high)
                ssim = calculate_ssim(pred, high)
                psnrs.append(psnr)
                ssims.append(ssim)

                if save_images:
                    save_root = os.path.join(self.val_root, str(epoch + 1))
                    os.makedirs(save_root, exist_ok=True)
                    save_path = os.path.join(save_root, f"{label}.png")
                    save_image(pred, save_path)

            psnr = sum(psnrs) / len(psnrs)
            ssim = sum(ssims) / len(ssims)
            self.logger.info(
                f"epoch: {epoch}, psnr: {psnr:.2f}, ssim: {ssim:.2f}")
            self.writer.add_scalar("val/PSNR", psnr, epoch)
            self.writer.add_scalar("val/SSIM", ssim, epoch)

            os.makedirs(self.ckpt_root, exist_ok=True)
            ckpts = os.listdir(self.ckpt_root)
            if psnr > self.best_score:
                self.best_score = psnr
                self.best_epoch = epoch
                best_ckpts = [f for f in ckpts if f.startswith("best")]
                self.save(f"best-{epoch}-{psnr:.2f}.pth")
                for f in best_ckpts:
                    os.remove(os.path.join(self.ckpt_root, f))
            last_ckpts = [f for f in ckpts if f.startswith("last")]
            self.save(f"last-{epoch}-{psnr:.2f}.pth")
            for f in last_ckpts:
                os.remove(os.path.join(self.ckpt_root, f))
        self.model.train(training)

    @property
    def out_dir(self):
        return getattr(self.config.training, "out_dir", None)

    @property
    def ckpt_root(self):
        if self.out_dir is not None:
            return self.out_dir
        return os.path.join(self.workspace, "ckpts")

    @property
    def workspace(self):
        if self.out_dir is not None:
            return self.out_dir
        return self.config.data.workspace

    @property
    def val_root(self):
        return os.path.join(self.workspace, "val")

    def save(self, name=None, epoch=None):
        epoch = self.epoch + 1 if epoch is None else epoch
        checkpoint = {
            "epoch": epoch,
            "step": self.step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema_helper": self.ema_helper.state_dict(),
            "params": self.args,
            "config": self.config
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()

        if name is None:
            name = str(epoch) + ".pth"
        checkpoint_path = os.path.join(self.ckpt_root, name)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        # write-then-rename so a kill mid-save never leaves a corrupt "latest"
        tmp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, checkpoint_path)

    def save_resumable(self):
        """Mid-training checkpoint; keeps the newest `keep_ckpts` (default 3)."""
        self.save(f"ckpt-{self.step:08d}.pth", epoch=self.epoch)
        keep = getattr(self.config.training, "keep_ckpts", 3)
        old = sorted(f for f in os.listdir(self.ckpt_root)
                     if f.startswith("ckpt-") and f.endswith(".pth"))
        for f in old[:-keep]:
            os.remove(os.path.join(self.ckpt_root, f))

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None, batch_size=8):
        steps = getattr(self.args, "sampling_timesteps", DDIM_STEPS)
        skip = self.config.diffusion.num_diffusion_timesteps // steps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        xs = diffusive_sample(
            x, x_cond, seq, self.model, self.betas, eta=0.,
            corners=patch_locs, p_size=patch_size,
            batch_size=batch_size,
        )
        return xs[0][-1] if last else xs

    def restore_single(self, cond, r=None, image_size=None, batch_size=8):
        cond = cond.unsqueeze(0)
        restored = self.restore(
            cond, r=r, image_size=image_size, batch_size=batch_size)
        restored = restored.squeeze(0)
        return restored

    def restore(self, x_cond, r=None, image_size=None, batch_size=8):

        p_size = self.config.data.image_size
        h, w = x_cond.shape[-2:]
        if image_size is not None:
            h, w = image_size

        corners = None
        if r is not None:
            # pad so the stride-r patch grid covers every pixel; cropped back below
            x_cond = pad_to_multiple(x_cond, math.lcm(p_size, r))
            h_list, w_list = overlapping_grid_indices(x_cond, p_size, r=r)
            corners = [(i, j) for i in h_list for j in w_list]
        else:
            from torch import nn
            model = self.model
            if isinstance(model, nn.DataParallel):
                model = model.module
            factor = model.scale
            h_x, w_x = x_cond.shape[-2:]
            pad_w = (w_x // factor + 1) * factor - w_x
            pad_h = (h_x // factor + 1) * factor - h_x
            from torch.nn.functional import pad
            x_cond = pad(x_cond, (0, pad_w, 0, pad_h), mode="reflect")

        x = torch.randn(x_cond.size(), device=self.device)
        x_cond = data_transform(x_cond)
        x_output = self.sample_image(
            x_cond, x, patch_locs=corners, patch_size=p_size, batch_size=batch_size)

        x_output = x_output[:, :, :h, :w]
        assert torch.isfinite(x_output).all(), "restored image contains NaN/inf"
        x_output = inverse_data_transform(x_output)
        return x_output
