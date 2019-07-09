
# InspectAI
InspectAI is a general purpose tool for inspecting manufacturing defects. Classic, rule based machine vision inspection requires re-programming for new parts or new defects, often struggles with abstract type of defect without clear geometric features or dimensions. InspectAI takes machine learning approach, specifically convolutional neural network, to achieve close to human level accuracy with consistency and scalability.

## Usecase
Multi-class multi-label problems.

## Dataset
Using a self captured and labeled image dataset of M&M which is an ideal object for this type of problem, showing various type of defects plus large quanity is easily accessbile. Of course this is just to prove the idea since I can't find any suitable and publicly available manufacturing defect dataset.

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

MultiLabel_newmm.ipynb ----> try out custom CNN and pre-trained CNN, tune threshold for recall-precision optimization

MultiLabel_CelebFaces.ipynb ----> test out the idea with CelebFacesAttributes dataset (multi-class, multi-label)

CropImage_GenerateLabel.ipynb ----> crop the center of images, clean the label file from LabelBox

crop_mm_cv2.ipynb ----> crop the images again around M&M

binary_mm.ipynb ----> reduce the problem to binary classification to see if accuracy improves

human_level_performance.ipynb ----> find human level accuracy, label same test set twice

AutoKeras_mm.ipynb ----> Try AutoKeras binary classification

autokeras_encode.ipynb ----> Relabel the dataset for AutoKeras multi-calss classification


