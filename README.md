# FER2013 — სახის გამომეტყველების ამოცნობა

Kaggle კომპეტიცია: [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)

WandB პროექტი: [fer2013](https://wandb.ai/ashos22-free-university-of-tbilisi-/fer2013)

GitHub: [fer2013-facial-expression](https://github.com/AvtandilSh1/fer2013-facial-expression)

---

## რეპოზიტორიის სტრუქტურა

```
fer2013-facial-expression/
├── arch1_tiny_cnn.ipynb           # არქიტექტურა 1 — TinyCNN
├── arch2_medium_cnn.ipynb         # არქიტექტურა 2 — MediumCNN
├── arch3_deep_cnn.ipynb           # არქიტექტურა 3 — DeepCNN
├── arch4_transfer_learning.ipynb  # არქიტექტურა 4 — MobileNetV2
└── README.md
```

---

## მონაცემთა ბაზა

FER2013 — 48x48 პიქსელის გრეისქეილ სურათები, 7 კლასი, ~35,887 ნიმუში.

| კლასი | ემოცია |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

---

## შედეგების შეჯამება

| არქიტექტურა | საუკეთესო Val Acc | შენიშვნა |
|---|---|---|
| TinyCNN (Adam, LR=1e-3) | 52.7% | Underfit — train=95.9%, val=52.7% |
| TinyCNN (Adam, LR=1e-4) | 39.5% | ძალიან ნელი კონვერგენცია |
| TinyCNN (SGD, LR=1e-2) | 52.0% | Underfit — train=99.6%, val=52.0% |
| MediumCNN — Dropout-ის გარეშე | 60.27% | Overfit — train=98.4%, val=60.3% |
| MediumCNN — Dropout=0.5 | 60.41% | Overfit მცირდება, val მსგავსი |
| MediumCNN — Dropout + Augmentation | 63.81% | საუკეთესო MediumCNN შედეგი |
| MediumCNN — Augment + Cosine + Class weights | 61.13% | Cosine LR-ი ამ შემთხვევაში ვერ დაეხმარა |
| DeepCNN — Adam LR=1e-3 | — | Run crashed (no summary saved) |
| DeepCNN — SGD + Cosine | 67.01% | საუკეთესო DeepCNN შედეგი |
| DeepCNN — Batch=128, LR=2e-3 | 66.17% | batch scaling — ოდნავ სუსტი |
| MobileNetV2 (Fine-tuned) | ~68% | საუკეთესო შედეგი |

---

## არქიტექტურული გადაწყვეტილებები

### არქიტექტურა 1: TinyCNN

მხოლოდ 2 conv layer, 8 და 16 ფილტრი. BatchNorm და Dropout არ არის.

მიზანი: განზრახ underfitting-ის საბაზო ხაზის დემონსტრაცია. მოდელს არ გააჩნია საკმარისი სიმძლავრე სასწავლო მონაცემების დასამახსოვრებლად.

შედეგი: train ~55%, val ~45%. პატარა სხვაობა train/val შორის = underfit (არა overfit). მოდელს არ ყოფნის სიმძლავრე და არა რეგულარიზაცია.

### არქიტექტურა 2: MediumCNN

4 conv layer (32→64→128→256 ფილტრი) + BatchNorm თითოეული conv-ის შემდეგ.

BatchNorm: ნორმალიზებს აქტივაციებს mini-batch-ის მიხედვით, დააჩქარებს სწავლებას.

ექსპერიმენტი A (Dropout-ის გარეშე): train=98.4%, val=60.27%. კლასიკური overfit — მოდელი ამახსოვრებს სასწავლო მონაცემებს. მიზეზი: დიდი FC head ~1.2M პარამეტრით, რეგულარიზაციის გარეშე.

ექსპერიმენტი B (Dropout=0.5): train=95.2%, val=60.41%. Dropout ამცირებს train accuracy-ს, val თითქმის იგივეა — overfit ჯერ კიდევ არის.

ექსპერიმენტი C (+Augmentation): train=66.1%, val=63.81%. RandomFlip, RandomRotation, RandomCrop — train/val სხვაობა მნიშვნელოვნად მცირდება, საუკეთესო MediumCNN შედეგი.

ექსპერიმენტი D (Augment + Cosine LR + Class weights): train=62.5%, val=61.13%. Cosine scheduler ამ კონფიგურაციაში C-ზე სუსტი — LR ნულამდე ჩამოდის და სწავლება ნაადრევად ჩერდება.

### არქიტექტურა 3: DeepCNN (Residual Blocks)

3-ეტაპიანი residual ქსელი Global Average Pooling-ით.

Residual კავშირები: `output = F(x) + x`. Gradient-ი პირდაპირ გადის skip connection-ით, ადრეული layer-ებიც იღებენ სათანადო gradient-ს. Backward pass-ის შემოწმება ამ ეფექტს ადასტურებს.

SGD + Cosine LR (67.01%) Adam-ს (crashed) სჯობდა ამ შემთხვევაში — SGD + momentum scratch-დან სწავლებულ CNN-ებში ხშირად უკეთეს გენერალიზაციას იძლევა. Batch=128 + LR=2e-3 (66.17%) ოდნავ ჩამოუვარდა — linear scaling rule-ი ყოველთვის არ მუშაობს პატარა datasets-ზე.

GAP MediumCNN-ის FC head-ის ნაცვლად:
- MediumCNN head: Flatten → FC(512) = 1.18M პარამეტრი
- DeepCNN head: GAP → Linear(512→7) = 3,591 პარამეტრი
- 330x ნაკლები პარამეტრი head-ში = ნაკლები overfit

### არქიტექტურა 4: TransferMobileNet (MobileNetV2)

ImageNet-ზე წინასწარ გავარჯიშებული MobileNetV2, FER2013-ზე ადაპტირებული.

პირველი conv ჩანაცვლდება 3-channel-იდან 1-channel-ზე. წინასწარი RGB წონები საშუალოდ ინიციალდება.

ორფაზიანი სწავლება:
- Phase 1: backbone გაყინულია, მხოლოდ head სწავლობს. მიზეზი: randomly initialized head-ს დიდი gradient-ები აქვს, რომლებიც pretrained წონებს გაანადგურებდა.
- Phase 2: LR=1e-4 (10x პატარა). pretrained წონები უკვე კარგ წერტილშია, დიდი ნაბიჯები გაანადგურებდა მათ.

კონტროლი (scratch-დან): MobileNetV2 pretrained weights-ის გარეშე ~63%, pretrained-ით ~68%. სხვაობა = transfer learning-ის სარგებელი.

---

## ჰიპერპარამეტრების ძიება

| არქიტექტურა | Optimizer | LR | Batch Size | Dropout | Scheduler |
|---|---|---|---|---|---|
| TinyCNN | Adam | 1e-3 | 64 | 0.0 | None |
| TinyCNN | Adam | 1e-4 | 64 | 0.0 | None |
| TinyCNN | SGD | 1e-2 | 64 | 0.0 | None |
| MediumCNN | Adam | 1e-3 | 64 | 0.0 | None |
| MediumCNN | Adam | 1e-3 | 64 | 0.5 | None |
| MediumCNN | Adam | 1e-3 | 64 | 0.5 | Cosine |
| MediumCNN | Adam | 1e-3 | 64 | 0.5 | Cosine + class weights |
| DeepCNN | Adam | 1e-3 | 64 | 0.4 | Cosine |
| DeepCNN | SGD | 1e-2 | 64 | 0.4 | Cosine |
| DeepCNN | Adam | 2e-3 | 128 | 0.4 | Cosine |
| MobileNetV2 | Adam | 1e-3 | 32 | 0.5 | Cosine (Phase 1) |
| MobileNetV2 | Adam | 1e-4 | 32 | 0.5 | Cosine (Phase 2) |
| MobileNetV2 scratch | Adam | 1e-3 | 32 | 0.5 | Cosine |

---

## Forward და Backward Pass შემოწმება

თითოეული სწავლების დაწყებამდე:

Forward pass: 4 random tensor გადის მოდელში. შემოწმება: output shape = (4,7), NaN/Inf არ არის.

Backward pass: ერთი dummy forward+backward pass. შემოწმება: ყველა layer-ი იღებს gradient-ს, gradient flow plot აჩვენებს ბალანსს.

არქიტექტურა 4 Phase 1-ისთვის: backbone layer-ებს gradient-ი არ აქვთ — ეს სწორია და მოსალოდნელი.

---

## WandB ლოგირების სტრუქტურა

თითოეული ექსპერიმენტი = ცალკე WandB run. ყველა run პროექტ `fer2013`-ში.

თითოეულ epoch-ზე:
- `train/loss`, `train/accuracy`
- `val/loss`, `val/accuracy`
- `grad_norm`
- `learning_rate`

Run-ის ბოლოს:
- `best_val_accuracy`
- confusion matrix
- run config
