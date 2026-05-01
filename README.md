# Modified DenseCL for Idiopathic Osteosclerosis (IO) Detection

Self-Supervised Learning framework berbasis **Modified DenseCL** dengan backbone **ResNet-50** dan downstream **Faster R-CNN** untuk deteksi Idiopathic Osteosclerosis pada citra panoramik gigi.

## Modifikasi DenseCL

Tiga modifikasi dilakukan pada DenseCL standar untuk mengatasi tantangan citra panoramik gigi:

### 1. Soft Lesion-Aware Weighting
- Dense contrastive loss diberi bobot lebih tinggi (`alpha=2.0`) pada region lesi
- Gaussian-smoothed boundary (`sigma=3.0`) untuk transisi gradual
- Konteks global tetap dipertahankan (background weight = 1.0)

### 2. Asymmetric Augmentation
- **View 1 (Global)**: Crop panoramik penuh sebagai global context
- **View 2 (Local)**: 50% di-bias ke region lesi dengan context padding 2× bounding box
- Pseudo-lesion detection berbasis intensity thresholding untuk unlabeled data

### 3. Radiologically-Informed Hard Negative Mining
- Region tulang normal yang mirip IO (cosine similarity + pixel intensity) dijadikan hard negative
- Warmup 20 epoch sebelum diaktifkan
- Combined scoring: 70% feature similarity + 30% intensity similarity

## Struktur Proyek

```
code/
├── configs/
│   └── config.py                    # Semua hyperparameter & ablation flags
├── data/
│   ├── data_filter.py               # 7-stage data filtering pipeline
│   ├── transforms.py                # Asymmetric augmentation & pseudo-lesion detector
│   ├── dataset_pretraining.py       # Dataset untuk SSL pretraining (unlabeled)
│   └── dataset_finetuning.py        # Dataset untuk Faster R-CNN (labeled, COCO format)
├── models/
│   ├── resnet.py                    # ResNet-50 backbone
│   ├── densecl_neck.py              # Dual-branch projection neck (global + dense)
│   ├── densecl_head.py              # InfoNCE contrastive head + hard negative support
│   ├── modified_densecl.py          # Core: Modified DenseCL dengan 3 modifikasi
│   └── faster_rcnn.py               # Faster R-CNN + FPN wrapper
├── utils/
│   ├── lesion_aware_weighting.py    # Modification 1: Soft lesion weighting
│   ├── hard_negative_mining.py      # Modification 3: Hard negative mining
│   └── metrics.py                   # mAP, IoU, precision/recall
├── scripts/
│   ├── filter_data.py               # CLI: 7-stage data filtering
│   ├── pretrain.py                  # CLI: SSL pretraining (ablation support)
│   ├── finetune.py                  # CLI: Faster R-CNN fine-tuning
│   ├── evaluate.py                  # CLI: Evaluasi mAP
│   └── extract_backbone.py          # CLI: Extract backbone weights
├── requirements.txt
└── README.md
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Pipeline

### Step 1: Data Filtering (50,000+ unlabeled images)

```bash
python scripts/filter_data.py \
    --input_dir /path/to/raw/panoramic/images \
    --output_dir data/unlabeled/filtered \
    --log_file output/filter_log.json
```

7 tahap filtering:
1. Resolusi minimum (800×400)
2. Aspect ratio panoramik (1.5–3.5)
3. Brightness range (30–220)
4. Contrast minimum (std ≥ 20)
5. Sharpness (Laplacian variance ≥ 10)
6. Dental domain check (histogram heuristic)
7. Deduplication (perceptual hash)

### Step 2: SSL Pretraining (Modified DenseCL)

```bash
# Full model (semua modifikasi aktif)
python scripts/pretrain.py \
    --data_dir data/unlabeled/filtered \
    --output_dir output/pretrain \
    --ablation full \
    --epochs 200 \
    --batch_size 32

# Baseline (vanilla DenseCL tanpa modifikasi)
python scripts/pretrain.py \
    --data_dir data/unlabeled/filtered \
    --ablation baseline \
    --epochs 200
```

### Step 3: Extract Backbone Weights

```bash
python scripts/extract_backbone.py \
    --checkpoint output/pretrain/full/checkpoints/best.pth \
    --output output/pretrain/full/backbone_weights.pth
```

### Step 4: Fine-tune Faster R-CNN (799 labeled images)

```bash
python scripts/finetune.py \
    --image_dir data/labeled/images \
    --annotation_file data/labeled/annotations.json \
    --backbone_weights output/pretrain/full/backbone_weights.pth \
    --output_dir output/finetune/full \
    --epochs 50 \
    --batch_size 4
```

### Step 5: Evaluate

```bash
python scripts/evaluate.py \
    --image_dir data/labeled/images \
    --annotation_file data/labeled/annotations.json \
    --checkpoint output/finetune/full/checkpoints/best.pth \
    --output_file output/eval_results.json
```

## Ablation Study

5 konfigurasi ablation tersedia:

| Config     | Mod 1 (Weighting) | Mod 2 (Asymmetric) | Mod 3 (Hard Neg) |
|-----------|:---------:|:----------:|:---------:|
| `baseline` | ✗ | ✗ | ✗ |
| `mod1`     | ✓ | ✗ | ✗ |
| `mod2`     | ✗ | ✓ | ✗ |
| `mod3`     | ✗ | ✗ | ✓ |
| `full`     | ✓ | ✓ | ✓ |

Jalankan pretraining untuk setiap konfigurasi:

```bash
for ablation in baseline mod1 mod2 mod3 full; do
    python scripts/pretrain.py \
        --data_dir data/unlabeled/filtered \
        --ablation $ablation \
        --epochs 200
done
```

## Format Anotasi (COCO JSON)

File anotasi untuk 799 labeled images harus dalam format COCO:

```json
{
    "images": [
        {"id": 1, "file_name": "panoramic_001.jpg", "width": 2000, "height": 1000}
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, width, height], "area": 500, "iscrowd": 0}
    ],
    "categories": [
        {"id": 1, "name": "IO"}
    ]
}
```

## Monitoring (TensorBoard)

```bash
tensorboard --logdir output/pretrain/full/tensorboard
```

## Hyperparameter Utama

| Parameter | Nilai Default | Keterangan |
|-----------|:---:|-----------|
| Global crop size | 256×512 | Landscape format untuk panoramik |
| Local crop size | 192×384 | Slightly smaller local crop |
| Queue length | 65,536 | Negative sample queue size |
| Feature dim | 128 | Embedding dimension |
| Momentum | 0.999 | EMA update untuk key encoder |
| Loss lambda | 0.5 | Weight global vs dense loss |
| Lesion alpha | 2.0 | Weight multiplier untuk region lesi |
| Gaussian sigma | 3.0 | Smoothing pada boundary lesi |
| Hard neg warmup | 20 epoch | Warmup sebelum hard neg aktif |
| Temperature | 0.2 | InfoNCE temperature |

## Referensi

- **DenseCL**: Wang et al., "Dense Contrastive Learning for Self-Supervised Visual Pre-Training", CVPR 2021
  - [Paper](https://arxiv.org/abs/2011.09157) | [Code](https://github.com/WXinlong/DenseCL)
- **MoCo v2**: Chen et al., "Improved Baselines with Momentum Contrastive Learning"
- **Faster R-CNN**: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
