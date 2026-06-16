# FER2013 — Facial Expression Recognition Challenge

Kaggle competition: [Challenges in Representation Learning: Facial Expression Recognition](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)

**WandB project:** `fer2013` — all runs logged there  
**Dataset:** 48×48 grayscale images, 7 emotions, ~35,887 samples

---

## Repository Structure

```
fer2013-facial-expression/
├── arch1_tiny_cnn.ipynb           # Architecture 1 — underfit baseline
├── arch2_medium_cnn.ipynb         # Architecture 2 — overfit → regularize
├── arch3_deep_cnn.ipynb           # Architecture 3 — residual blocks
├── arch4_transfer_learning.ipynb  # Architecture 4 — MobileNetV2
└── README.md
```

---

## Results Summary

| Architecture | Best Val Acc | Notes |
|---|---|---|
| TinyCNN | ~45% | Underfit — too small |
| MediumCNN (no dropout) | ~58% | Overfit — train>>val |
| MediumCNN (regularized) | ~62% | Dropout + augment |
| DeepCNN (residual) | ~65% | Residual blocks + GAP |
| MobileNetV2 fine-tuned | ~68% | Best overall |

---

## Architecture Decisions

### Architecture 1: TinyCNN

**Decision:** 2 conv layers with 8 and 16 filters. No BatchNorm, no Dropout.

**Why:** Establishing an intentional underfit baseline. A model that cannot even fit training data shows us the minimum capacity threshold needed.

**Result:** Train ~55%, Val ~45%. Small train/val gap confirms underfitting (not overfitting). The model lacks capacity, not regularization.

**What we learned:** Need at least 4 conv layers and BatchNorm to train stably on this task.

---

### Architecture 2: MediumCNN

**Decision:** 4 conv layers (32→64→128→256 filters) + BatchNorm after each conv.

**Why BatchNorm:** Normalizes activations per mini-batch → faster convergence, acts as mild regularization, allows higher learning rates.

**Experiment A (no Dropout):** Train reaches 80-85%, Val plateaus at ~58%. This is textbook overfitting — the model memorized training data instead of learning general features. Root cause: 4 wide layers with ~1.2M parameters in the FC head and no regularization.

**Experiment B (Dropout=0.5):** Single change. The train/val gap shrinks significantly. Dropout forces the network to use redundant features → less memorization.

**Experiment C (+ augmentation):** Each epoch the model sees different variations of each image (flipped, rotated, cropped). Effectively multiplies training data diversity → further improvement.

**Experiment D (+ class weights + cosine LR):** FER2013 is imbalanced (Happy has 3× more samples than Disgust). Class weights penalize errors on minority classes more. Cosine LR prevents oscillation in the final training phase.

**Remaining problem:** The FC head (`Flatten → 2304 → FC(512)`) still has too many parameters. This is the main remaining source of overfit.

---

### Architecture 3: DeepCNN (Residual Blocks)

**Decision:** 3-stage residual network with Global Average Pooling.

**Why residual connections:** Without skip connections, deep networks suffer from vanishing gradients — gradient magnitude decreases exponentially with depth. The residual formula `output = F(x) + x` provides a direct gradient highway from output to early layers. Verified in backward pass check: stem layer gradients are similar magnitude to classifier gradients.

**Why Global Average Pooling instead of Flatten+FC:**
- MediumCNN head: `Flatten → 2304 → FC(512)` = **1.18M parameters**
- DeepCNN head: `GAP → Dropout → Linear(512→7)` = **3,591 parameters**
- 330× fewer head parameters → dramatically less overfit in the classification layer

**Why Dropout only in the final layer (0.4):** The convolutional backbone is already regularized by BatchNorm and augmentation. Heavy dropout in conv layers hurts performance on small datasets.

**Batch size experiment:** Doubling batch_size from 64→128 requires scaling LR by ~2× (linear scaling rule) to maintain equivalent gradient step size. Larger batches converge faster per epoch but may find sharper loss minima.

---

### Architecture 4: TransferMobileNet (MobileNetV2)

**Why MobileNetV2:** Pretrained on 1.2M ImageNet images. Early layers learned universal visual features (edges, textures, shapes) that transfer well to faces. MobileNetV2 chosen over ResNet50 for efficiency (3.4M vs 25M params) given Kaggle GPU constraints.

**Grayscale adaptation:** MobileNetV2's first conv expects 3-channel input. We replace it with a 1-channel conv and initialize it by averaging the 3 pretrained RGB weight channels. This is better than random initialization — we preserve the learned edge detectors.

**Why two-phase training:**
- Phase 1 (frozen backbone): The new classifier head is randomly initialized → produces large gradients in early training. If we unfreeze the backbone immediately, these large gradients would corrupt the pretrained features. Training only the head first stabilizes it.
- Phase 2 (LR=1e-4): 10× smaller learning rate than Phase 1. Pretrained weights are already near a good solution — large LR steps would destroy them. Small steps gently adapt backbone features to the FER2013 domain.

**Control experiment (no pretrained weights):** MobileNetV2 architecture trained from scratch achieved ~63%, vs ~68% with pretraining. The ~5% gap is the direct value of ImageNet pretraining on this task.

---

## Forward & Backward Pass Checks

Performed before every training run:

**Forward pass:** Feeds 4 random tensors through the model, verifies:
- Output shape is `(4, 7)` — one logit per class
- No NaN or Inf values

**Backward pass:** One dummy forward + backward pass, verifies:
- All trainable layers receive gradients (no dead layers)
- Gradient flow plot shows gradients are balanced across depth
- For Architecture 3: residual connections visibly equalize gradient magnitudes

**Exception for Architecture 4 Phase 1:** Backbone layers intentionally have NO gradient (they are frozen). This is the expected and correct behavior.

---

## WandB Logging Structure

Each experiment = one WandB run. All runs in project `fer2013`.

Logged per epoch:
- `train/loss`, `train/accuracy`
- `val/loss`, `val/accuracy`
- `grad_norm` — backward pass health monitor
- `learning_rate` — tracks scheduler behavior

Logged per run (summary):
- `best_val_accuracy`
- Confusion matrix (WandB Table)
- Run config (architecture, lr, dropout, batch_size, etc.)
