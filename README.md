Skeletal Muscle Radiomics Worklow - Code Implementation

This repository contains the core scripts used to run the radiomics workflow for distinguishing children diagnosed with cerebral palsy from typically developing children. This includes feature extraction using PyRadiomics, a feature selection and ranking pipeline, a statistical analysis using a Mann-Whitney U test, and classification using RadiomiX. An additional script demonstrating how MuscleMap segmentation can be performed is also included. The goal is to allow easy reproducibility and future radiomic analysis using a similar workflow. 

No medical data is or should be included in this repository. The provided code can be implemented on medical data of different types but may have to be altered depending on data format.


## Design: This project is intentionally split into different stages with clear inputs and outputs.

01. **Feature extraction (PyRadiomics)**
   - Script: `01_feature_extraction.py`
   - Input: Image and segmentation mask with correct format and same spatial geometry. If not already done, another script converting mask to the same spatial geometry should be done beforehand.
   - Output: Extracted feature values. 
   Note: This script currently shows a framework for how feature extraction should be used for one image and mask. In practice, this is easiest converted to a loop going though several images+masks and saving the extracted feature values to a CSV for further use.  

02. **Feature selection and ranking pipeline**
   - Script: `02_feature_selection_and_ranking_pipeline.py`
   - Input: Multiple CSV files of feature values from two different subject groups, multiple muscles and sexes.
   - Output: Most important features averaged across folds.
   Note: The script is organized after one CSV naming convention where features are extracted from subject types, image contrast, muscles, and sex individually. This can differ by study and may therefore need to be altered.

03. **Statistical significance analysis**
   - Script: `03_statistical_significance.py`
   - Input: 10 most important features from script 02 from two subject groups for group comparison.
   - Output: p-values following Mann-Whitney U test.

04. **RadiomiX model evaluation**
   - Script: `04_classification.py`
   - Input: 10 most important features from script 02 from two subject groups for group classification.
   - Output: Folder containing performance results for different RadiomiX model combinations. This includes averaged AUC, accuracy and F1 scores as well as best performing parameters.
   Note: RadiomiX scripts were edited to account for this dataset and reduced computation costs. 

Extra scripts: 
  **Seperate segmentation stage**
   - Script: `MuscleMap_segmentation.py`
   - Input: Image with correct format. 
   - Output: Generated segmentation mask corresponding to input image. 

  **Custom RadiomiX workflow**
   - Script: `radiomix_edit.py`
   - Note: RadiomiX workflow adjusted to account for data used in this study. Use instead of radiomix.py inside of RadiomiX folder.

  **Custom parameter combination**
   - Script: `custom_params.json`
   - Note: Parameter combination search space adjusted for this study.



## How to Run ##

This code currently uses script level configuration blocks, (constants at the top of each script). Make sure all file paths and inputs are correct before running.

The workflow requires the installation of PyRadiomics, RadiomiX, and MuscleMap. These packages are open source and publically availabel to download from GitHub.

1: Make sure to have images and segmented masks with the same spatial geometry so that PyRadiomics can extract features in 01.
2: Use 01 framework, expand to loop across subjects and save to CSV files for individual cases such as one subject group and muscle. 
3: Manually enter necessary inputs based on CSV format and naming convention used when running 01, thereafter run either 03 and 04 to get results. When running 04, define which model combinations to use in `custom_params.json`. 
X: Try MuscleMap segmentation using XX.  
