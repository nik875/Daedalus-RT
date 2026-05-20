"""
Daedalus attack reimplemented in PyTorch targeting YOLO26's NMS-free one2one head.

Single-image digital attack (equivalent to the original l2_yolov3.py): optimise
a full-image perturbation for one specific image so YOLO26 emits a flood of
high-confidence spurious detections.

Key implementation notes:
  - YOLO26's one2one head detaches its feature inputs in Detect.forward to
    prevent gradient interference during model training.  We patch this out
    so gradients flow back to the input for the attack.
  - Loss: the paper's confidence push — mean((sigmoid(score) - 1)^2) over all
    slots and classes (l2_yolov3.py loss1_1_x).
    No w*h term — without NMS there is nothing to exploit with tiny boxes.
  - Optimizer: mSAM (Sharpness-Aware Minimization) wrapping AdamW, with a
    linear-warmup + cosine-decay learning-rate schedule.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage import io
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect


# ---------------------------------------------------------------------------
# Patch YOLO26 Detect head
# ---------------------------------------------------------------------------
# The one2one head detaches its feature inputs by design (prevents gradient
# interference between heads during model training). For the attack we need
# gradients to flow from the one2one scores all the way back to the image.

_original_detect_forward = Detect.forward


def _patched_detect_forward(self, x):
    # Always return the raw head predictions dict (pre-NMS, no postprocess),
    # independent of self.training.  This lets us keep the whole model in
    # eval() mode — so BatchNorm uses stable running stats and avoids the
    # cuDNN training-mode batch_norm path — while still exposing the one2one
    # scores the attack needs.  The one2one head's inputs are NOT detached
    # (unlike the stock head) so gradients flow back to the input image.
    preds = self.forward_head(x, **self.one2many)
    if self.end2end:
        one2one = self.forward_head(x, **self.one2one)
        preds = {"one2many": preds, "one2one": one2one}
    return preds


Detect.forward = _patched_detect_forward


# ---------------------------------------------------------------------------
# mSAM optimizer
# ---------------------------------------------------------------------------

class mSAM:
    """
    Micro-batch Sharpness-Aware Minimization.

    Each optimisation step requires two forward+backward passes:
      1. ascent_step(): perturb params in the direction of steepest gradient
                        ascent, scaled to L2 ball of radius rho.
      2. descent_step(): compute gradient at the perturbed point, restore
                         original params, apply base_optimizer update.

    With batch_size=1 this is identical to standard SAM.  With larger batches
    (e.g. EOT), ascent is computed per-example so each example gets its own
    perturbation direction rather than following the mean gradient.
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, **base_kwargs):
        self.params = list(params)
        self.base_optimizer = base_optimizer_cls(self.params, **base_kwargs)
        self.rho = rho
        self._saved_e = {}

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return torch.tensor(1e-12)
        return torch.norm(torch.stack([g.norm(2) for g in grads]))

    def ascent_step(self):
        """Perturb params toward steepest ascent (SAM neighbourhood)."""
        norm = self._grad_norm() + 1e-12
        for p in self.params:
            if p.grad is None:
                continue
            e_w = (self.rho / norm) * p.grad.detach()
            p.data.add_(e_w)
            self._saved_e[id(p)] = e_w

    def descent_step(self):
        """Restore params then apply base optimizer using current gradients."""
        for p in self.params:
            buf = self._saved_e.pop(id(p), None)
            if buf is not None:
                p.data.sub_(buf)
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Daedalus attack
# ---------------------------------------------------------------------------

# Defaults
LEARNING_RATE    = 3e-3
ITERATIONS       = 1000
# loss = adv_weight * adv_loss + l2_weight * mean_per_pixel_L2.
# Both terms are normalised to order ~1, so the weights are directly comparable.
# l2_weight << adv_weight prioritises the attack while lightly penalising
# visible distortion (the paper's CW binary search effectively did this by
# driving its trade-off constant c very large).
ADV_WEIGHT       = 1.0
L2_WEIGHT        = 0.05
SAM_RHO          = 0.025
IMAGE_SIZE       = 640
MODEL_PATH       = "yolo26n.pt"
SAVE_PATH        = "adv_examples/yolo26/"

# Universal-perturbation defaults
EPOCHS           = 10
BATCH_SIZE       = 8
EPSILON          = 16 / 255   # L-inf budget on the universal perturbation
UNIVERSAL_SAVE_PATH = "adv_examples/yolo26_universal/"


class Daedalus:
    """
    Single-image Daedalus attack targeting YOLO26's one2one head.

    Optimises a full-image perturbation for one specific image that drives
    every one2one slot's confidence toward 1.0 (the paper's loss), filling
    YOLO26's output with high-confidence spurious boxes.

    Loss      : adv_weight * confidence_loss + l2_weight * L2 distortion
    Optimizer : mSAM (SAM) wrapping AdamW
    Schedule  : linear warmup (first 10%) then cosine decay
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        learning_rate=LEARNING_RATE,
        iterations=ITERATIONS,
        adv_weight=ADV_WEIGHT,
        l2_weight=L2_WEIGHT,
        rho=SAM_RHO,
        device=None,
    ):
        self.lr = learning_rate
        self.iterations = iterations
        self.adv_weight = adv_weight
        self.l2_weight = l2_weight
        self.rho = rho
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        yolo = YOLO(model_path)
        self.model = yolo.model.to(self.device)
        # Keep the whole model in eval() so BatchNorm uses stable pretrained
        # running stats (matching deployment) and avoids the cuDNN train-mode
        # batch_norm path.  The patched Detect.forward returns the head dict
        # regardless of training mode, so we don't need train() here.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scores(self, newimgs):
        """Return sigmoid class scores from the one2one head: (N, 80, 8400)."""
        out = self.model(newimgs)
        return torch.sigmoid(out["one2one"]["scores"])

    def _adv_loss(self, scores):
        """
        Paper's confidence loss: mean squared push of every score toward 1,
        averaged over all slots and classes (l2_yolov3.py loss1_1_x).
        """
        return torch.mean((scores - 1.0) ** 2)

    def _top_score(self, scores):
        """Mean of the top-300 scores — display-only progress metric."""
        return scores.reshape(scores.shape[0], -1).topk(300, dim=1).values.mean().item()

    def _l2_dist(self, newimgs, orig):
        """Per-pixel mean squared distortion (order ~1, comparable to adv_loss)."""
        return torch.mean((newimgs - orig) ** 2)

    def _to_img(self, w_orig, delta):
        """tanh reparameterisation keeps pixels in [0, 1]."""
        return torch.tanh(w_orig + delta) * 0.5 + 0.5

    # ------------------------------------------------------------------
    # Core attack
    # ------------------------------------------------------------------

    def _attack_single(self, img_np):
        """
        Attack one image.

        img_np: (H, W, 3) float32 in [0, 1]
        returns: (H, W, 3) float32 adversarial image
        """
        orig = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )  # (1, 3, H, W)

        # arctanh-space representation of the original image
        w_orig = torch.arctanh((orig * 2.0 - 1.0).clamp(-0.999999, 0.999999))

        delta = torch.zeros_like(w_orig, requires_grad=True)
        optimizer = mSAM([delta], torch.optim.AdamW, rho=self.rho, lr=self.lr)
        warmup_steps = max(1, int(self.iterations * 0.1))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer.base_optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer.base_optimizer,
                    start_factor=0.1, end_factor=1.0, total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer.base_optimizer,
                    T_max=self.iterations - warmup_steps, eta_min=self.lr * 0.01,
                ),
            ],
            milestones=[warmup_steps],
        )

        pbar = tqdm(range(self.iterations), desc="attack", leave=True, ascii=True)
        for _ in pbar:
            # --- SAM ascent: first forward pass ---
            optimizer.zero_grad()
            newimgs = self._to_img(w_orig, delta)
            loss = self.adv_weight * self._adv_loss(self._scores(newimgs)) \
                + self.l2_weight * self._l2_dist(newimgs, orig)
            loss.backward()
            optimizer.ascent_step()

            # --- SAM descent: second forward pass at perturbed point ---
            optimizer.zero_grad()
            newimgs = self._to_img(w_orig, delta)
            scores = self._scores(newimgs)
            adv_loss = self._adv_loss(scores)
            l2 = self._l2_dist(newimgs, orig)
            loss = self.adv_weight * adv_loss + self.l2_weight * l2
            loss.backward()

            grad_norm = delta.grad.norm().item() if delta.grad is not None else 0.0
            top_score = self._top_score(scores.detach())
            optimizer.descent_step()
            scheduler.step()

            pbar.set_postfix(
                adv=f"{adv_loss.item():.4f}",
                top300=f"{top_score:.4f}",
                l2=f"{l2.item():.2e}",
                grad=f"{grad_norm:.2e}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        final = self._to_img(w_orig, delta)
        return final.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    def attack(self, imgs, save_path=SAVE_PATH):
        """
        Attack one or more images independently and save results.

        imgs: list of (H, W, 3) float32 numpy arrays in [0, 1]
        returns: (N, H, W, 3) numpy array of adversarial images
        """
        os.makedirs(save_path, exist_ok=True)
        results = []
        distortions = []

        for i, img in enumerate(imgs):
            print(f"\n=== Image {i + 1}/{len(imgs)} ===")
            adv = self._attack_single(img)
            l2 = float(np.sum((adv - img) ** 2))
            print(f"  Final L2 distortion: {l2:.4f}")

            results.append(adv)
            distortions.append(l2)
            io.imsave(
                os.path.join(save_path, f"adv_{i:04d}_l2={l2:.3f}.png"),
                (adv * 255).clip(0, 255).astype(np.uint8),
            )

        results = np.array(results)
        np.savez(
            os.path.join(save_path, "batch.npz"),
            X_adv=results,
            distortions=np.array(distortions),
        )
        return results


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SceneDataset(torch.utils.data.Dataset):
    """Loads all images from a directory, resized to a fixed size, RGB [0, 1]."""

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, image_dir, size=IMAGE_SIZE):
        self.size = size
        self.paths = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(image_dir)
            for f in files
            if os.path.splitext(f)[1].lower() in self.EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Universal perturbation
# ---------------------------------------------------------------------------

class DaedalusUniversal(Daedalus):
    """
    Universal adversarial perturbation against YOLO26's one2one head.

    Trains a single full-image delta over a dataset so that adding it to ANY
    image floods the output with high-confidence spurious boxes.

    Perturbation is additive and constrained to an L-inf epsilon ball
    (standard UAP formulation), applied as clamp(img + delta, 0, 1).
    Inherits the model setup and loss helpers from Daedalus.
    """

    def train(
        self,
        image_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        epsilon=EPSILON,
        save_path=UNIVERSAL_SAVE_PATH,
        checkpoint_every=1,
    ):
        os.makedirs(save_path, exist_ok=True)

        dataset = SceneDataset(image_dir)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=(self.device == "cuda"), drop_last=True,
        )

        # The universal perturbation — one delta shared across every image.
        delta = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE,
                            device=self.device, requires_grad=True)
        optimizer = mSAM([delta], torch.optim.AdamW, rho=self.rho, lr=self.lr)
        total_steps = epochs * len(loader)
        warmup_steps = max(1, int(total_steps * 0.1))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer.base_optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer.base_optimizer,
                    start_factor=0.1, end_factor=1.0, total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer.base_optimizer,
                    T_max=total_steps - warmup_steps, eta_min=self.lr * 0.01,
                ),
            ],
            milestones=[warmup_steps],
        )

        print(
            f"\n=== Training universal perturbation (eps={epsilon:.4f}) ===\n"
            f"    dataset : {len(dataset)} images  |  batch : {batch_size}"
            f"  |  epochs : {epochs}\n"
        )

        for epoch in range(1, epochs + 1):
            ep_adv = ep_top = 0.0
            n_steps = 0
            pbar = tqdm(loader, desc=f"epoch {epoch:3d}/{epochs}", leave=True, ascii=True)
            for imgs in pbar:
                imgs = imgs.to(self.device)

                # mSAM ascent
                optimizer.zero_grad()
                loss = self.adv_weight * self._adv_loss(
                    self._scores((imgs + delta).clamp(0.0, 1.0)))
                loss.backward()
                optimizer.ascent_step()

                # mSAM descent
                optimizer.zero_grad()
                scores = self._scores((imgs + delta).clamp(0.0, 1.0))
                adv = self._adv_loss(scores)
                loss = self.adv_weight * adv
                loss.backward()
                grad_norm = delta.grad.norm().item() if delta.grad is not None else 0.0
                top = self._top_score(scores.detach())
                optimizer.descent_step()
                scheduler.step()

                # Project delta back into the L-inf epsilon ball
                with torch.no_grad():
                    delta.clamp_(-epsilon, epsilon)

                ep_adv += adv.item(); ep_top += top
                n_steps += 1
                pbar.set_postfix(
                    adv=f"{adv.item():.4f}",
                    top300=f"{top:.4f}",
                    grad=f"{grad_norm:.2e}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            print(f"  -> epoch {epoch:3d} | adv={ep_adv / n_steps:.4f} "
                  f"top300={ep_top / n_steps:.4f}")
            if epoch % checkpoint_every == 0:
                self._save_delta(delta, save_path, tag=f"epoch{epoch:03d}")

        return self._save_delta(delta, save_path, tag="final")

    def _save_delta(self, delta, save_path, tag=""):
        """Save the raw perturbation (.npy) and a viewable normalised PNG."""
        d = delta.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        suffix = f"_{tag}" if tag else ""
        np.save(os.path.join(save_path, f"delta{suffix}.npy"), d)
        # Normalise to [0, 1] for visualisation (delta is small and signed)
        vis = (d - d.min()) / (d.max() - d.min() + 1e-12)
        io.imsave(os.path.join(save_path, f"delta{suffix}.png"),
                  (vis * 255).clip(0, 255).astype(np.uint8))
        return d


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(path, size=IMAGE_SIZE):
    """Load, resize to model input size, and normalise to [0, 1]."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "universal"], default="single")
    parser.add_argument("--image-dir", default="../Datasets/COCO/val2017/")
    # single mode
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    # universal mode
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--epsilon",    type=float, default=EPSILON)
    args = parser.parse_args()

    if args.mode == "universal":
        DaedalusUniversal().train(
            args.image_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
        )
    else:
        imgs = []
        for root, _, files in os.walk(args.image_dir):
            for f in sorted(files):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                imgs.append(preprocess(os.path.join(root, f)))
                if len(imgs) >= args.num_images:
                    break
            if len(imgs) >= args.num_images:
                break
        if not imgs:
            raise FileNotFoundError(f"No images found in {args.image_dir}")

        Daedalus(iterations=args.iterations).attack(imgs)
