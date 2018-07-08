# Behavioral-Cloning
This repository contains the codes and the outputs for the Udacity Self driving car nanodegree program term1-project3. This repository contains files model.py, model.h5, drive.py, run1.mp4 and video.py. **model.py** contains the model as well as the other pre-processing techniques used on the images before they are used for training. **drive.py** contains the code for running the simulator with the trained model in the Autonomous mode of the simulator provided by Udacity. **run1.mp4** contains the output of the car driving in the autonomous mode. The simulator along with the datasets obtained are not included in the repository. **model.h5** contains the trained model weights. This can be used only for track 1 and it has not been tested for track 2. 

## Dataset
The dataset for the behavioral cloning project were obtained using the simulator provided by Udacity in the training mode. The images were considered as training inputs and the steering angles were considered as the output values. In this case, the deep learning network was trained for a regression purpose. The input to the model will be an image and the output will be the steering angle that has to be sent to the car to stay on the track provided. 

## Data Visualization and Pre-Processing
Once that dataset was obtained using the simulator, the dataset can be visualized to understand how the images are fed to the model. 
