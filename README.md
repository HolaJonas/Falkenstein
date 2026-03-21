# Falkenstein

A **DenseNet121**-based classification model for identifying 200 different bird species from the **CUB-200-2011** dataset.

## Overview

**Falkenstein** achieves strong performance in bird species classification using transfer learning from ImageNet pre-trained weights. The model was developed during a 3-day hackathon and demonstrates the effectiveness of **DenseNet121** for fine-grained visual classification tasks.

## Dataset

**CUB-200-2011** (Caltech-UCSD Birds-200-2011):
- **200 bird species** with multiple images per species
- **~11,788 images** total images

CUB-200-2011 dataset should be placed in **/data**. 

Download from: https://data.caltech.edu/records/65de6-vp158. 
You may use **src.falkenstein.data.prepare_dataset** to unpack the tgz-file.

## Model Architecture

**DenseNet121** with custom classification layer:
- Pre-trained ImageNet weights
- Configurable classifier hidden dimensions

See [models.py](src/falkenstein/models.py) for implementation details.

## Results

| Metric | Score |
|--------|-------|
| Train Accuracy | 99.6% |
| Validation Accuracy | 84.14% |
| Test Accuracy | 82.3% |
| Top-5 Test Accuracy | 96% |

**Note:** The model exhibits overfitting (validation/test accuracy ~15% lower than training). Improvements are still possible.


## Usage

### Training

After adding your **config** and **dataset**, run the model training pipeline:

```bash
python main.py
```

**Configuration file** (`configs/config.yaml`) parameters:
- `train_split`: Training set fraction (0.0-1.0)
- `validation_split`: Validation set fraction (0.0-1.0)
- `test_split`: Test set fraction (0.0-1.0)
- `learning_rate`: AdamW learning rate
- `label_smoothing`: Label smoothing factor (0.0-1.0)
- `patience`: Early stopping patience (epochs)
- `num_epochs`: Maximum training epochs
- `batch_size`: Batch size for training
- `weight_decay`: L2 regularization
- `classifier_hidden_dim`: Hidden dimension of classifier head
- `output_layers_type`: Type of output layers. "mlp" or "linear"
- `dropout`: Dropout probability

The existing [config](configs/config.yaml) yielded best results.
### Evaluation

Test the trained model:
- Use [test](src/falkenstein/test.py) for evaluation
- Pre-trained weights stored in [weights](weights/) directory


## Status

**Project Status**: Complete (Hackathon submission)

This project was developed as a 3-day informal hackathon submission. While the model achieves competitive results, further improvements are possible but limited by computational resources during development. The codebase is stable and serves as a solid foundation for future bird classification work.

## Citation

If using this project or CUB-200-2011 dataset, please cite:

```bibtex
@misc{wah_branson_welinder_perona_belongie_2022, title={CUB-200-2011}, DOI={10.22002/D1.20098}, abstractNote={CUB-200-2011 is an extended version of CUB-200, a challenging dataset of 200 bird species. The extended version roughly doubles the number of images per category and adds new part localization annotations. All images are annotated with bounding boxes, part locations, and attribute labels. Images and annotations were filtered by multiple users of Mechanical Turk.}, publisher={CaltechDATA}, author={Wah, Catherine and Branson, Steve and Welinder, Peter and Perona, Pietro and Belongie, Serge}, year={2022}, month={Apr} }
```