# CS5330 Computer Vision Final Project: Satellite Image Classification System

Using Vision Transformer (ViT) model to classify the land use and land cover from satellite images. The model is pretrained on ImageNet and then fine-tuned on EuroSat dataset with validation accuracy of 97.58% and test accuracy of 97.62. Due to the limited GPU, the best model can only be fine-tuned based on the following parameters (searched by Optuna):
- Learning rate: 2.5342001417793284e-05
- batch_size: 64
- unfreeze: 3
- weight_decay: 0.01

## Features

- **FastAPI backend** for model inference
- **Streamlit web interface** for easy interaction
- **Pre-trained ViT model** fine-tuned on EuroSAT dataset
- Supports classification of 10 land cover classes:
  - Annual Crop 🌾
  - Forest 🌲
  - Herbaceous Vegetation 🌿
  - Highway 🛣️
  - Industrial 🏭
  - Pasture 🐄
  - Permanent Crop 🍇
  - Residential 🏘️
  - River 🌊
  - Sea/Lake 🌊

  ## Usage

  Make sure to have vit_eurosat_best_model.pth under api folder

  ```
  python api/api_service.py
  ```

  ```
  streamlit run app/streamlit_app.py
  ```
