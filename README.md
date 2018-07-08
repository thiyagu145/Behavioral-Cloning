# Behavioral-Cloning
This repository contains the codes and the outputs for the Udacity Self driving car nanodegree program term1-project3. This repository contains files model.py, model.h5, drive.py, run1.mp4 and video.py. **model.py** contains the model as well as the other pre-processing techniques used on the images before they are used for training. **drive.py** contains the code for running the simulator with the trained model in the Autonomous mode of the simulator provided by Udacity. **run1.mp4** contains the output of the car driving in the autonomous mode. The simulator along with the datasets obtained are not included in the repository. **model.h5** contains the trained model weights. This can be used only for track 1 and it has not been tested for track 2. 

## Dataset
The dataset for the behavioral cloning project were obtained using the simulator provided by Udacity in the training mode. The images were considered as training inputs and the steering angles were considered as the output values. In this case, the deep learning network was trained for a regression purpose. The input to the model will be an image and the output will be the steering angle that has to be sent to the car to stay on the track provided. The dataset is divided into testing and training data using the test_train_split provided in the sklearn.utils library. 

## Data Visualization and Pre-Processing
Once that dataset was obtained using the simulator, the dataset can be visualized to understand how the images are fed to the model. Data augmentation is also done on random to reduced over-fitting. This is done by flipping the images in random. The original images along with their steering angles are in column 1 and flipped images along with their inverted steering angles are given in column 2 of the following image. 
![alt text](https://github.com/thiyagu145/Behavioral-Cloning/blob/master/Data%20Visualization.png)

As a second augmentation step, the images from all three cameras namely left, right and center are also considered in random. While the image batches are created for training, the images are chosen in random among the three camera images. There is a correction factor added to the steering angles when the left and the right camera images are taken. A correction of +0.2 and -0.2 are added to the steering angles for the left and the right camera images respectively. This is done so that the model considers these images as images from the centre camera. A distribution of the camera angle values is given below. 

![alt text](https://github.com/thiyagu145/Behavioral-Cloning/blob/master/Steering%20angles%20distribution.png)

After the data is augmented, the images have to be pre-processed. The first pre-processing step used is to crop the images. From the visualization of the images we can see that the top and bottom part of the images are basically sky and the hood of the car respectively. The model does not need these information for determining the steering angle and hence these portions of the images can be discarded. After the cropping is done, the images are resized to (66,200) from (160,320) since it is the size of the input image for the NVIDIA self driving car architecture. These are the two pre-processing techniques used. Once the images are resized, the images are normalized between the range of (-1,1) to prevent the vanishing gradient problem.  

## Model Architecture
The Deep learning network architecuture used in this project is a modified and more simpler version of the NVIDIA self driving car control architecture. The architecture is given below:

![alt text](https://github.com/thiyagu145/Behavioral-Cloning/blob/master/Architecture.png)

The architecture used in this is project is also very similar, except that a subsampling of size (2,2) is used to reduce the complexity of the model. The original NVIDIA architecture contains about 60 million parameters whereas this reduced model contains only 1.5 million parameters. This greatly reduces the training times at the same time the accuracy is also high.

Details of the Model:
1) Number of convolution layers: 5 
2) Activation function used: 'relu'
3) Number of epochs: 3
4) Dropout layers: 4
5) Optimizer used: Adam 

Dropout layers are used before each of the dense layers and also after the first convolution layer. The dropout layers help us prevent over-fitting of the model towards the training data. 

## Model performance
The model took about 20 minutes for each epoch on amazon AWS g2.2xlarge gpu instance. The model performs very well on the first track. The video of the car on the first track is present in the file **run1.mp4**. One some curves, the car touches the lane markings but at other places, the car drives very well by itself. If the original NVIDIA architecture is used, the car will definitely be able to autonomously drive better on the roads. But this is a very good trade-off between the performance and the model complexity, as the model used in this project has almose 40 times lesser complexity than the original NVIDIA model. 


