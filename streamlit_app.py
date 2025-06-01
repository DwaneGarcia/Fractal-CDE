import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
import models
import torch

from torchvision import transforms
from torchvision import transforms

def load_model(path, model):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

def predict(img):
    model = models.unet(3, 1)
    model = load_model('model.pth',model)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    img = cv2.resize(img, (512, 512))
    convert_tensor = transforms.ToTensor()
    img =  convert_tensor(img).float()
    img = normalize(img)
    img = torch.unsqueeze(img, dim=0)

    output = model(img)
    result = torch.sigmoid(output)

    threshold = 0.5
    result = (result >= threshold).float()
    prediction = result[0].cpu()  # Move tensor to CPU if it's on GPU
    # Convert tensor to a numpy array
    prediction_array = prediction.numpy()
    # Rescale values to the range [0, 255]
    prediction_array = (prediction_array * 255).astype('uint8').transpose(1, 2, 0)
    cv2.imwrite("test.png",prediction_array)
    return prediction_array

def predicjt(img):
    model1 = models.SAunet(3, 1)
    model1 = load_model('saunet.pth',model1)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    img = cv2.resize(img, (512, 512))
    convert_tensor = transforms.ToTensor()
    img =  convert_tensor(img).float()
    img = normalize(img)
    img = torch.unsqueeze(img, dim=0)

    output = model1(img)
    result = torch.sigmoid(output)

    threshold = 0.5
    result = (result >= threshold).float()
    prediction = result[0].cpu()  # Move tensor to CPU if it's on GPU
    # Convert tensor to a numpy array
    prediction_array = prediction.numpy()
    # Rescale values to the range [0, 255]
    prediction_array = (prediction_array * 255).astype('uint8').transpose(1, 2, 0)
    cv2.imwrite("test1.png",prediction_array)
    return prediction_array
def main():
    st.title("Image Segmentation Demo")

    # Predefined list of image names
    image_names = ["01_test.tif", "02_test.tif", "03_test.tif"]

    # Create a selection box for the images
    selected_image_name = st.selectbox("Select an Image", image_names)

    # Load the selected image
    selected_image = cv2.imread(selected_image_name)

    # Display the selected image
    st.image(selected_image, channels="RGB")

    # Create a button for segmentation
    if st.button("Segment"):
        # Perform segmentation on the selected image
        segmented_image = predict(selected_image)
        segmented_image1 = predicjt(selected_image)


        # Display the segmented image
        st.image(segmented_image, channels="RGB",caption='U-Net segmentation')
        st.image(segmented_image1, channels="RGB",caption='Spatial Attention U-Net segmentation ')

# Function to perform segmentation on the selected imagee


if __name__ == "__main__":
    main()
