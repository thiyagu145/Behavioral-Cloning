# Behavioral-Cloning
This repository contains the codes and the outputs for the Udacity Self driving car nanodegree program term1-project3. This repository contains files model.py, model.h5, drive.py, run1.mp4 and video.py. **model.py** contains the model as well as the other pre-processing techniques used on the images before they are used for training. **drive.py** contains the code for running the simulator with the trained model in the Autonomous mode of the simulator provided by Udacity. **run1.mp4** contains the output of the car driving in the autonomous mode. The simulator along with the datasets obtained are not included in the repository. **model.h5** contains the trained model weights. This can be used only for track 1 and it has not been tested for track 2. 

## Dataset
The dataset for the behavioral cloning project were obtained using the simulator provided by Udacity in the training mode. The images were considered as training inputs and the steering angles were considered as the output values. In this case, the deep learning network was trained for a regression purpose. The input to the model will be an image and the output will be the steering angle that has to be sent to the car to stay on the track provided. 

## Data Visualization and Pre-Processing
Once that dataset was obtained using the simulator, the dataset can be visualized to understand how the images are fed to the model. Data augmentation is also done on random to reduced over-fitting. This is done by flipping the images in random. The original images along with their steering angles are in column 1 and flipped images along with their inverted steering angles are given in column 2 of the following image. 
![alt text](https://github.com/thiyagu145/Behavioral-Cloning/blob/master/Data%20Visualization.png)

As a second augmentation step, the images from all three cameras namely left, right and center are also considered in random. While the image batches are created for training, the images are chosen in random among the three camera images. There is a correction factor added to the steering angles when the left and the right camera images are taken. A correction of +0.2 and -0.2 are added to the steering angles for the left and the right camera images respectively. This is done so that the model considers these images as images from the centre camera. A distribution of the camera angle values is given below. 

![alt text](https://github.com/thiyagu145/Behavioral-Cloning/blob/master/Steering%20angles%20distribution.png)

After the data is augmented, the images have to be pre-processed. The first pre-processing step used is to crop the images. From the visualization of the images we can see that the top and bottom part of the images are basically sky and the hood of the car respectively. The model does not need these information for determining the steering angle and hence these portions of the images can be discarded. Once the cropping is done, 

