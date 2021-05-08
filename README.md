#  Enhanced U-Net: A Feature Enhancement Network for Polyp Segmentation

## Requirements

* torch
* torchvision
* scipy
* PIL
* numpy
* tqdm

### Data
```
$ data
train
├── Images
├── Masks
valid
├── Images
├── Masks
test
├── Images
├── Masks
```

### 1. Training

```bash
python train.py  --mode train  --train_data_dir /path-to-train_data  --valid_data_dir  /path-to-valid_data
```

####  2. Testing

```bash
python test.py  --mode test  --load_ckpt checkpoint --test_data_dir  /path-to-test_data```

