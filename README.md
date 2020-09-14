# Kaokore Social Status Prediction

Identifying Social Statuses in Historical Japanese Artwork.

In this project we will use ResNet to identify the social statuses of faces in pre-modern Japanese artwork. The cropped facial images are collected from multiple Japanese artworks. The goal is to classify each facial image into one of the four social statuses, i.e., noble, warrior, incarnation, and commoner. The example images of each class are shown as follows:

![Kaokore social status](https://github.com/NaeRong/Kaokore_SocialStatus_Prediction/blob/master/Prediction/Classes.png)

In order to complete the prediction model, we will follow the steps below:

- **Data Preparation**
  - Download the KaoKore dataset from https://github.com/rois-codh/kaokore by following the instructions in the Github repo. The
    KaoKore dataset includes 5,552 images with size 256×256. There are two sets of labels: gender and social status. In this project, we only consider the social status.
  - **Load the Data with PyTorch** When loading the data, use functions in torchvision.transforms to conduct data augmentation and normalization. Also using *Resize* to reduce the
    size of input images to save GPU memory and accelerate training.
    
- **Network Setting / Training**
  - Use ResNet Model for training. Model parameters used are the following:
    * Batch size : 32
    * Number of epochs: 50
    * Learning rate : 0.001
    * Optimizer: Adam 
    * Criterion: CrossEntropyLoss
    * Activation Function: ReLU
    
    For more details on the ResNet model, please visit this Python file:
    [**ResNet Model**](https://github.com/NaeRong/Kaokore_SocialStatus_Prediction/blob/master/Prediction/ResNet.py)
    
- **Network Testing**
  - Test the accuracy on the testing samples. The model accuracy rates are:
    * **Train** : 88.15%
    * **Validation** : 76.17%
    * **Test** : 77.80%
    
    The Training and validation loss graph is shown as below:
    
    ![Train and Validation Loss](https://github.com/NaeRong/Kaokore_SocialStatus_Prediction/blob/master/Prediction/Train_Vali_Loss.png)
    
    The example of image prediction results on the test dataset:
    
    ![Test / Validation / Train Prediction Results](https://github.com/NaeRong/Kaokore_SocialStatus_Prediction/blob/master/Prediction/Prediction_Test.png)
    
    


  
