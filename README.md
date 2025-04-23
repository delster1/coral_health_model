# Coral Health Model

## Project Structure
coral_health_model/
- data/ - all images for training
    - images-flouro/ - segmented coral (flouro)
    - images-non-flouro/ - ...
    - masks-flouro/ - region-identifying masks (flouro)
    - masks-non-flouro/ - ...
- models/ - our models for segmentation & classification
- dataset/ - dataset functions
- train/ - scripts for training 
- utils/ - preprocessing - create training data & visualize output

## Implementation
Using U-Net's Segmentation Model to perform pixel-wise classification of coral images to identify regions as healthy, healing, or bleached.

We have four options for training the model, which are as follows:
1. Training with flouro images processed in grayscale.
2. Training with flouro images processed in full-color.
3. Training with non-flouro images processed in grayscale.
4. Training with non-flouro images processed in full-color.

We aim to determine a model's ability to identify regions of coral as healing, healthy, or bleached.

## Training 

The model is trained with two pieces of data:
1. The original image of the coral, either in grayscale or full-color, showing the natural structure and features of the coral head.
2. A corresponding segmentation mask - an image of the same size that encodes pixel-level class labels. Each pixel in the mask represents the health status of that part of the coral using discrete values:
- `0` -> Healthy Region 
- `1` -> Healing Region
- `2` -> Bleached Region 
- `255` -> Ignore Index / Background - pixels not used for loss calculation.

The segmentation masks are manually created by overlaying color-coded annotations onto the coral images. During preprocessing, they are converted to numerical label maps, where each color is mapped to it's corresponding index.

## Preprocessing / Augmentation 

We use light augmentations during training including:
- Random horizontal/vertical flips
- Small rotations (-30 to +30 degrees)
- Optional color conversion to grayscale

Masks are cleaned with morphological operations to remove speckling and converted into class index maps before training.

## Evaluation
We evaluate segmentation performance using:
- Pixel-wise accuracy
- Per-class IoU (Intersection over Union)
- Visual comparison of predictions vs masks

Future work may include Dice score and region-level metrics for ecological impact.


## Installation & Usage
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/coral_health_model.git
   cd coral_health_model
    ```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate
```

3. Install Requirements

```bash
pip install -r requirements.txt
```

### Usage
To train the model, run the model, and visualize predictions:
```bash
python3 run_training --config=configs/flouro.yaml
```

## Sources
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation [arXiv preprint arXiv:1505.04597]. https://arxiv.org/abs/1505.04597

CodezUp. (2020). Image segmentation using U-Net in PyTorch. https://codezup.com/a-hands-on-guide-to-image-segmentation-using-u-net-in-pytorch/

PyTorch. (n.d.). PyTorch documentation (stable). https://pytorch.org/docs/stable/index.html

Wikipedia contributors. (2023, March 30). Image segmentation. Wikipedia. https://en.wikipedia.org/wiki/Image_segmentation
