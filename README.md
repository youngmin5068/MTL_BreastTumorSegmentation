# InterpreSegNet


Multi-task learning model for breast tumor segmentation and classification in breast MRI.


## Model Architecture

![Model_architecture](https://github.com/youngmin5068/InterpreSegNet/assets/61230321/538c69c2-039b-46a9-97c8-2fa21eb7a898)


This multi-task learning approach simultaneously performs classification and segmentation tasks, sharing an encoder between them.
The classifier not only identifies tumors but also predicts their approximate locations. 
The shared learning process enhances the segmentation performance. 
By focusing on slices identified as containing tumors, this method enables more precise tumor localization on a per-slice basis, improving the accuracy of tumor location identification.
