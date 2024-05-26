# MTL Breast Tumor Segmentation


Multi-task learning model for breast tumor segmentation and classification in breast MRI.


## Model Architecture

![Model_architecture](https://github.com/youngmin5068/InterpreSegNet/assets/61230321/538c69c2-039b-46a9-97c8-2fa21eb7a898)


This multi-task learning approach simultaneously performs classification and segmentation tasks, sharing an encoder between them.
The classifier not only identifies tumors but also predicts their approximate locations. 
The shared learning process enhances the segmentation performance. 
By focusing on slices identified as containing tumors, this method enables more precise tumor localization on a per-slice basis, improving the accuracy of tumor location identification.

## Topt-CBAM 

### Channel Attention Module of Topt-CBAM
<img width="650" alt="스크린샷 2024-05-23 오후 12 40 01" src="https://github.com/youngmin5068/MTL_BreastTumorSegmentation/assets/61230321/b571769c-1b99-494a-93b3-4548a719a9f5">

Topt-CBAM is a variant of the Channel Attention Module in CBAM. 
In the Channel Attention Module, it pooled only the top t% of pixel values for max pooling and average pooling to focus more on the important information.


## Third Party Library

torch==1.11.0
albumentations==1.3.1
einops==0.7.0
glob2==0.7
monai==1.3.0
numpy==1.24.3
opencv-python==4.9.0.80
openpyxl==3.1.2
pandas==2.0.3
pillow==9.0.1
pydicom==2.4.4
pylibjpeg-libjpeg==2.0.0
pylibjpeg==2.0.0
scipy==1.10.1
simpleitk==2.3.1
timm==0.9.12
