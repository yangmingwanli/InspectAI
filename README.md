
# InspectAI
InspectAI is a general purpose tool for inspecting manufacturing defects. Classic, rule based machine vision inspection requires re-programming for new parts or new defects, often struggles with abstract type of defect without clear geometric features or dimensions. InspectAI takes machine learning approach, specifically convolutional neural network, to achieve close to human level accuracy with consistency and scalability.

## Usecase
Supervised multi-class multi-label image classification problems. Ideal for inspecting manufacturing defects that are too abstract to explicitly program for using classic rule based machine vision.

## Model
Different model architures are explored and benchmarked against human performance (labeling the same test dataset twice myself). For this type of inspection, custom CNN is the overall best option in terms of close to human level accuracy, short training time, fast inference speed and light model size.

![Alt text](analysis/models.png?raw=true "Title")

The custom CNN is a sequence of linear and non-linear matrix operations, starting with low level simple feature detectors of edges and dots, distilling into high level abstract features of letter 'M', shapes and surfaces etc. As a black box, it maps an input image into possiblities of having various types of defects.

![Alt text](analysis/customCNN.png?raw=true "Title")

## Dataset
Using a self captured and labeled image dataset of M&M which is an ideal object for this type of problem, showing various type of defects plus large quanity is easily accessbile. Of course this is just to prove the idea since I can't find any suitable and publicly available manufacturing defect dataset.

1336 M&Ms images are taken with smartphone camera, preprocessed with OpenCV and labeled with LabelBox in four mostly indepentdent categories (partiall missing letter 'M', off center letter 'M', non circular shape, surface blemish). 

![Alt text](examples/mm.png?raw=true "Title")

## Try it out

Clone the repo
```bash
git clone https://github.com/yangmingwanli/InspectAI.git
```
Download the images
```bash
cd <data folder of repo>
wget https://1336mms.s3-us-west-2.amazonaws.com/mm.zip
unzip mm.zip
```
Train the model
```bash
python inspectAI/train/mm_train.py
```
Deploy the model
```bash
python inspectAI/deploy/run_keras_server.py
```
Inference (update IP address of host in script first)
```bash
vi inspectAI/deploy/simple_request.py
python inspectAI/deploy/simple_request.py
```
## Experiment
Experiment folder contains these jupyter notebooks used for experiments and data pre-processing

1. MultiLabel_newmm.ipynb
    - try out custom CNN and pre-trained CNN, tune threshold for recall-precision optimization
    
2. MultiLabel_CelebFaces.ipynb 
    - test out the idea with CelebFacesAttributes dataset (multi-class, multi-label)
    
3. CropImage_GenerateLabel.ipynb
    - crop the center of images, clean the label file from LabelBox

4. crop_mm_cv2.ipynb
    - crop the images again around M&M

5. binary_mm.ipynb 
    - reduce the problem to binary classification to see if accuracy improves

6. human_level_performance.ipynb 
    - find human level accuracy, label same test set twice

7. AutoKeras_mm.ipynb 
    - Try AutoKeras binary classification

8. autokeras_encode.ipynb 
    - Relabel the dataset for AutoKeras multi-calss classification

## Project Overview

[Link on Youtube](https://www.youtube.com/watch?v=QpYNQia9pW4)
