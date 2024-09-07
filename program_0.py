# In this Python Streamlit application we will show how to use LIME framework
# with Pytorch. Specifically, we will use LIME to explain predictions made by
# one of the (selectable) pretrained ImageNet models in Pytorch.


# Let's start by importing the dependencies.

import io, json, os

import streamlit as st

from PIL import Image

import torch, torchvision
from torchvision import models, transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pandas as pd


st.title('Lime Image Classification')

st.write('''
Intuitively, an explanation is a local linear approximation of the model's behaviour. While the model may be very complex globally, it is easier to approximate it around the vicinity of a particular instance. While treating the model as a black box, we perturb the instance we want to explain and learn a sparse linear model around it, as an explanation. The figure below illustrates the intuition for this procedure. The model's decision function is represented by the blue/pink background, and is clearly nonlinear. The bright red cross is the instance being explained (let's call it X). We sample instances around X, and weight them according to their proximity to X (weight here is indicated by size). We then learn a linear model (dashed line) that approximates the model well in the vicinity of X, but not necessarily globally.
''')
st.image('dogs.png')


# Upload a test image and see how it looks.

st.subheader('Select Image')

def get_image(bytes_data):
    with io.BytesIO(bytes_data) as f:
        with Image.open(f) as img:
            return img.convert('RGB') 
        
uploaded_file = st.file_uploader('Choose an image file',
                                 type=['png', 'jpg', 'jpeg'])
if uploaded_file is None:
    st.stop()
bytes_data = uploaded_file.getvalue()
img =  get_image(bytes_data)
st.image(img)


# We need to convert this image to Pytorch tensor and also apply whitening as
# used by our pretrained model.

def get_input_transform():
    '''Resize and take the center part of image to what our model expects'''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])       
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])    

    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)


# Load the pretrained model names and let the user select the model to be used.

st.subheader('Select Classification Model')

torchvision_models = models.list_models(module=torchvision.models)
model_name = st.selectbox('Which classification model would you like to use?',
                          torchvision_models)
st.write(f'The selected classification model is {model_name}.')

# Load the selected model.

model = models.get_model(model_name, weights='DEFAULT')
st.write(model)

from torchsummary import summary
summary(model, (3, 256, 256))


# Load label texts for ImageNet predictions so we know what model is
# predicting.

idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('./data/imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}


#  Get the prediction for our image.

img_t = get_input_tensors(img)
model.eval()
logits = model(img_t)


# Predicitions we got are logits. Let's pass that through softmax to get
# probabilities for top 5 predictions.

probs = F.softmax(logits, dim=1)
probs5 = probs.topk(5)

# Display the top 5 predictions in a Pandas data frame.
st.subheader('Top five (5) predictions')
d = {'probability': probs5[0][0].detach().numpy(),
     'index': probs5[1][0].detach().numpy(),
     'label': [idx2label[c] for c in probs5[1][0].detach().numpy()]}
df = pd.DataFrame(data=d)
st.dataframe(df)


# We are getting ready to use Lime. Lime produces the array of images from the
# original input image by pertubation algorithm. So we need to provide two
# things: (1) original image as numpy array (2) classification function that
# would take array of purturbed images as input and produce the probabilities
# for each class for each image as output.

# For Pytorch, first we need to define two separate transforms: (1) to take PIL
# image, resize and crop it (2) take resized, cropped image and apply whitening.

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)])    
    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize])    
    return transf    

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()


# Now we are ready to define classification function that Lime needs. The input
# to this function is numpy array of images where each image is ndarray of shape
# (channel, height, width). The output is numpy aaray of shape (image index,
# classes) where each value in array should be probability for that image, class
# combination.

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


# Let's test our function for the sample image.
# test_pred = batch_predict([pill_transf(img)])
# test_pred.squeeze().argmax()


# Import lime and create explanation for this prediciton.

from lime import lime_image

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                         batch_predict, # classification function
                                         top_labels=5, 
                                         hide_color=0, 
                                         num_samples=1000) # number of images that will be sent to classification function


# Let's use mask on image and see the areas that are encouraging the selected
# prediction.

from skimage.segmentation import mark_boundaries

idx = st.number_input('Which prediction (0-4) should be explained?',
                      min_value = 0, max_value=4, value=0)

col1, col2 = st.columns(2)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[idx],
                                            positive_only=True,
                                            num_features=5,
                                            hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)

with col1:
    st.image(img_boundry1)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[idx],
                                            positive_only=False,
                                            num_features=5,
                                            hide_rest=False)
img_boundry2 = mark_boundaries(temp/255.0, mask)

with col2:
    st.image(img_boundry2)
