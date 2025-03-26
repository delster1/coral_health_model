# Coral Health Model

## Project Structure
coral_health_model/
- data/ - all images for training
    - images-flouro/ - segmented coral (flouro)
    - images-non-flouro/ - ...
    - masks-flouro/ - region-identifying masks (flouro)
    - maske-non-flouro/ - ...
- models/ - our models for segmentation & classification
- dataset/ - dataset functions
- train/ - scripts for training 
- utils/ - preprocessing - create training data & visualize output
