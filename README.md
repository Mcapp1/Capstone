# Capstone: Classifying CT Scans of Lung Cancer
### Michael Capparelli

## Problem Statement:
Accurately classify chest computed tomography scans containing various forms of lung cancer such as adenocarcinomic, large cell carcinoma, squamous cell carcinoma, and other malignant cases. This is a binary problem which uses deep learning neural nets to determine whether a ct scan contains cancer or not.


## Background:
The American Cancer Society’s estimates for lung cancer in the United States for 2022 are:
- About 236,740 new cases of lung cancer (117,910 in men and 118,830 in women)
- About 130,180 deaths from lung cancer (68,820 in men and 61,360 in women)

Lung Cancer, the leading cause of cancer related deaths and second most common cancer, makes up almost 25% of all cancer related deaths. This is more than colon, breast, and prostate cancer related deaths combined. Chest x-rays were long used as a screening test for lung cancer however in recent years, low-dose CT scans have been studied as a more common practice. Regular chest x-rays haven't been shown to help most people live longer, and therefore aren't recommended for lung cancer screenings. However CT scans can help find abnormal areas in the lungs that may be cancerous. Research has shown that unlike chest x-rays, yearly CT scans can save lives specifically in high risk patients. For these patients, getting yearly CT scans before symptoms start helps lower the risk of dying from lung cancer.

https://www.cancer.org/cancer/lung-cancer/about/key-statistics.html
https://www.cancer.gov/types/common-cancers#:~:text=The%20most%20common%20type%20of,are%20combined%20for%20the%20list.

## Data Sources
Several thousand images were aggregated from the following:
- https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset
    - This dataset is used as a main source of non-cancerous ct scans. The dataset consists of ~7000 COVID ct scans and ~7000 non-covid scans. No COVID images were used, only non-covid/normal scans as to compile enough data to train on. Several normal scans may not have contained COVID however contained other anomolies, or the patient was holding their breath. Those pictures were not included. 
    - Additionally this dataset contained duplicate images under different file names, I used roughly 800 images for my non-cancerous data.
- https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
    - Roughly 500 images were used from this dataset. The dataset includes ct scans of adenocarcinomic, large cell carcinoma, and squamous cell carcinomic cancers. After manually inspecting all images, I noticed there were several duplicates under different file names. Duplicates were not included to the best of my abilities.
    - Additionally this dataset is pre-labeled. All 3 types of cancer were aggregated as one to keep this as a binary problem.
- https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset
    - This dataset contains 561 malignant cases and 400 normal cases, again containing duplicate images under different file names. After sorting through these as well, I used roughly 400 images for my cancer data and filled the rest of my non-cancerous data.
- https://www.cancerimagingarchive.net/
    - The last of the cancer images, under 100 images were used from this dataset.   
### All data has been unzipped, converted, stored, and labeled in the data directory as JPG and PNG images under train, valid, and test data containing 2000, 400, and 80 images respectively.

## Executive Summary









