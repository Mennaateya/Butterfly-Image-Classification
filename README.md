# Butterfly Image Classification

Predict the species of butterflies from images using **CNN from scratch** and **ResNet50V2 fine-tuning**.

**You can try it now:** https://butterfly-image-classification.streamlit.app/

## Dataset

- **Source:** [Kaggle Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)
- **Training images:** `Training_set.csv` with labels  
- **Testing images:** `Testing_set.csv` (to predict labels)  
- Total classes: 75  
- Each image belongs to a single butterfly species.

## Preprocessing

- Images resized to 224×224.  
- Data augmentation: rotation, width/height shift, shear, zoom, horizontal flip.  
- `preprocess_input` from ResNet50V2 used for normalization.  
- Split training set into training (90%) and validation (10%).

## Models Trained

- **Custom CNN:** 5 convolutional layers with batch normalization, max pooling, dropout, and global average pooling.  
- **Transfer Learning (ResNet50V2):** pretrained on ImageNet, last 20 layers fine-tuned, added dense layers, batch normalization, and dropout.  

Evaluation metrics: Accuracy, Loss, Visual inspection of predictions.

## Results

| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| Custom CNN | 0.76 | 0.861 |
| ResNet50V2 Fine-tuned | 1.00 | 0.0065 |

> Fine-tuning pretrained ResNet50V2 significantly outperforms the CNN trained from scratch.

## Visualizations

- Sample images with true and predicted labels.  
- Training vs validation loss and accuracy curves.  
- Comparison of CNN vs ResNet50V2 predictions.

## Usage

1. Clone the repository.  
2. Install dependencies (`tensorflow`, `keras`, `pandas`, `numpy`, `matplotlib`).  
3. Download dataset from Kaggle.  
4. Run the notebook to train models and evaluate performance.

## Files

- `Training_set.csv` → Training image labels  
- `Testing_set.csv` → Testing image names  
- `train/` → Training images  
- `test/` → Testing images  
- `butterfly_model_deeper_100epochs.keras` → Custom CNN trained model  
- `pretrained_model.keras` → ResNet50V2 fine-tuned model  
