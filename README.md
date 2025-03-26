# Coral Health Model

## Project Structure
coral_health_model/
- data/ - all images for training
    - images-flouro/ - segmented coral (flouro)
    - images-non-flouro/ - segmented coral (non-flouro)
    - masks-flouro/ - region-identifying masks (flouro)
    - maske-non-flouro/ - region-identifying masks (non-flouro)
- models/ - our models for segmentation & classification
- dataset/ - maybe a custom pytorch dataset
- train/ - scripts for training 
- utils/ - preprocessing - create training data & visualize output
