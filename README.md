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
| TinyCNN | ~45% | Underfit |
| MediumCNN (Dropout-ის გარეშე) | ~58% | Overfit |
| MediumCNN (რეგულარიზებული) | ~62% | Dropout + Augmentation |
| DeepCNN (Residual) | ~65% | Residual Blocks + GAP |
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

ექსპერიმენტი A (Dropout-ის გარეშე): train ~80-85%, val ~58%. ტექსტური overfit — მოდელი ამახსოვრებს სასწავლო მონაცემებს. მიზეზი: დიდი FC head ~1.2M პარამეტრით, რეგულარიზაციის გარეშე.

ექსპერიმენტი B (Dropout=0.5): train/val სხვაობა მნიშვნელოვნად მცირდება.

ექსპერიმენტი C (+Augmentation): RandomFlip, RandomRotation, RandomCrop — კიდევ უმჯობესდება.

ექსპერიმენტი D (+Class weights + Cosine LR): FER2013 არაბალანსირებულია (Happy 3x მეტია Disgust-ზე). Class weights უმცირეს კლასებს ეხმარება.

### არქიტექტურა 3: DeepCNN (Residual Blocks)

3-ეტაპიანი residual ქსელი Global Average Pooling-ით.

Residual კავშირები: `output = F(x) + x`. Gradient-ი პირდაპირ გადის skip connection-ით, ადრეული layer-ებიც იღებენ სათანადო gradient-ს. Backward pass-ის შემოწმება ამ ეფექტს ადასტურებს.

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

