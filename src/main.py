from train import train_pipeline
from predict import predict_image
import os

if __name__ == "__main__":
    # Train the model and get the configuration and trainer
    cfg, trainer = train_pipeline()

    # Read an image from the local file system for prediction
    image_path = input("Enter the path of the image you want to predict: ")

    # Predict on the uploaded image using the trained model and configuration
    predict_image(cfg, image_path)
