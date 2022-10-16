# Capstone: Classifying CT Scans of Lung Cancer
### Michael Capparelli

## Problem Statement:
Accurately classify chest computed tomography scans containing various forms of lung cancer such as Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, and other malignant cases. This is a binary problem that uses deep-learning neural nets to determine whether a CT scan contains cancer or not.


## Background:
The American Cancer Society’s estimates for lung cancer in the United States for 2022 are:
- About 236,740 new cases of lung cancer (117,910 in men and 118,830 in women)
- About 130,180 deaths from lung cancer (68,820 in men and 61,360 in women)

Lung Cancer, the leading cause of cancer-related deaths and the second most common cancer behind breast cancer, make up almost 25% of all cancer-related deaths. This is more than colon, breast, and prostate cancer-related deaths combined. 

Chest x-rays were long used as a screening test for lung cancer however in recent years, low-dose CT scans have been studied as a more common practice. Regular chest x-rays haven't been shown to help most people live longer, and therefore aren't recommended for lung cancer screenings. However, CT scans have shown to find abnormal areas in the lungs that may be cancer. Research has shown that, unlike chest x-rays, yearly CT scans can save lives specifically in high-risk patients. For these patients, getting yearly CT scans before symptoms start helps lower the risk of dying from lung cancer.

https://www.cancer.org/cancer/lung-cancer/about/key-statistics.html
https://www.cancer.gov/types/common-cancers#:~:text=The%20most%20common%20type%20of,are%20combined%20for%20the%20list.

## Data Sources
Several thousand images were aggregated from the following:
- https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset
    - This dataset is used as the main source of non-cancerous CT scans. The dataset consists of ~7000 COVID CT scans and ~7000 non-covid scans. No COVID images were used, only non-covid/normal scans to compile enough data to train on. Several normal scans may not have contained COVID however contained other anomalies. Those pictures were not included. 
    - Additionally, this dataset contained duplicate images under different file names, I used roughly 800 images from this dataset for my non-cancerous data.
- https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
    - Roughly 500 images were used from this dataset. The dataset includes CT scans of Adenocarcinoma, Large Cell Carcinoma, and Squamous Cell Carcinoma cancers. After manually inspecting all images, I noticed there were several duplicates under different file names. Again, duplicates were not included to the best of my ability.
    - Furthermore, this dataset is pre-labeled. All 3 types of cancer were aggregated as one to keep this as a binary problem.
- https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset
    - This dataset contains 561 malignant cases and 400 normal cases, again containing duplicate images under different file names. After examination of the data, I used roughly 400 images for my cancer data and filled in the rest of my non-cancerous data.
- https://www.cancerimagingarchive.net/
    - The last of the cancer images, under 100 images were used from this dataset.   
#### All data has been unzipped, converted, stored, and labeled in the data directory as JPG and PNG images under train, valid, and test data containing 2000, 400, and 80 images respectively.

## Executive Summary

Healthcare is a demanding field where effeciency and timely response is can be the difference between life and death. For this project, I aggregated, cleaned, and labeled 2480 images of chest CT scans. This is a binary problem where the scan either contains cancer or doesn't. Splitting each class in my training, validation and testing sets equally, a baseline of 50% is established. Constructing convolutional neural networks while optimizing for high accuracy, recall and minimizing false negatives is the goal. This is because it would be better to tell a patient they have cancer when they do not compared to telling someone they do not have cancer when they do. After constructing several custom models and implementing transfer learning models with my own regularization methods, I gather the following results.


|Model|Accuracy|Cancer Recall|Non-Cancer Recall|
|-----|--------|----------------|--------------------|
|Custom|95%|97%|93%|
|ResNet50|99%|97%|100%|
|VGG16|99%|97%|100%|
|MobileNetV2|94%|88%|100%|

Since all images are axial slices of a chest CT scan, the ResNet50 model is most ideal to use not only due to outperforming the baseline by 47% but because ResNet50 operates better with grayscale images than the other models. Additionally the one image whcih was misclassified as a false negative was misclassified in all models which led me to believe there was an error in pre-processing with this specific image.

The purpose of this model is to demonstrate the precision and efficiency that deep learning brings to the healthcare field. You should never base a diagnosis on what a model says. The real-world purpose is for a physician to use a model such as this as a reference. By eliminating cases where both a doctor and the model is confident the patient does not have cancer, they can focus on cases they believe do have cancer. Doing so could save lives and using CT scans has already proven to be the case!

## Data Dictionary

|Name|Description|Required?|Data Type|
|----|-----------|---------|---------|
|train_path|denotes path to all training images|Y|Directory|
|test_path|denotes path to all testing images|Y|Directory|
|valid_path|denotes path to all validation images|Y|Directory|
|display_sample|displays 9 labeled images in a 3x3 block of both cancer and non cancer images|N|Function|
|train_gen|pre-processing training images|Y|ImageDataGenerator|
|test_gen|pre-processing testing images|Y|ImageDataGenerator|
|valid_gen|pre-processing validation images|Y|ImageDataGenerator|
|image_distribution|plots the image count of each class to whichever dataset is called|N|Function|
|model_4|tensorflow keras custom model|Y|keras.engine.functional.Functional|
|model_results|classification report, confusion matrix, and misclassified images of whichever model is called|Y|Function|
|build_resnet50|function which builds the ResNet50 model|Y|Function|
|resnet_model|tf and keras transfer learning model|Y|keras.engine.functional.Functional|
|plot_loss|displays training and validation loss functions of models history|N|Function|
|build_vgg116|function which builds the VGG16 model|Y|Function|
|vgg16_model|tf and keras transfer learning model|Y|keras.engine.functional.Functional|
|build_mobilenet|function which builds the MobileNetV2 model|Y|Function|
|mobilenet_model|tf and keras transfer learning model|Y|keras.engine.functional.Functional|
|predict_any_image|upload any image to be predicted on specifically the ResNet50 model|Y|Function|

## Conclusions

The original problem statement asks if it is possible to accurately and precisely classify whether a CT scan displays cancer or not. After constructing many models, I can conclude that it is possible. The ResNet50 model answers this question best, bolstering an accuracy of .99 on the entirety of the testing data as well as a recall of .97 and 1.0 for cancer and non-cancer images respectively. A focus was to minimize false negatives, and the ResNet50 model only misclassified one image. This one image was recurring through all models which leads me to believe there may be an issue with how the image is pre-processed.

## Recommendations

- From here, I would mask and segment the lungs. Doing so would allow for a better idea of where cancer could be in the lungs. I attempted to do this however due to time constraints I was unable to complete this task.
- High frequency in file repetition caused a lot of time to be spent looking over the images. To keep the train-validation split at an acceptable rate I ended up having only 80 images to test my model on. I would gather more data and test how my model performs on larger datasets.
- Additionally to segmenting the lungs, training the model to localize where it believes cancer is would be ideal as well. 