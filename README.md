# EITSync: An Out-of View Hand Tracking Method Utilizing Electrical Impedance Tomography for VR and AR


## Installation 
We recommend configuring the project inside an Anaconda environment. We have tested everything using [Anaconda](https://docs.anaconda.com/anaconda/install/) version 23.9.0 and Python 3.9. The first step is to create a virtual environment, as shown below (named `EITSync`).
```
conda create -n EITSync python=3.9
```
You should then activate the environment as shown below. All following operations must be completed within the virtual environment.
```
conda activate mobileposer
```
Then, install the required packages.
```
pip install -r requirements.txt
```
You will then need to install the local mobileposer package for development via the command below. You must run this from the root directory (e.g., where setup.py is).
```
pip install -e .
```
For any further missing packages, simply run 
```
conda install [package name]
```
# Running Files

To run files, run them from the EITSync directory like
```
python mobileposer/[nameoffile].py
```

# Data Visualizer
The data visualizer works for visualizing ground truth data as well as predicted data.
The necessary unity project for data visualization can be found [here](https://drive.google.com/file/d/1Zx1mU-K7MQH0bvkGrm7nF_CYnFeveQAO/view?usp=sharing)
To run the data visualizer, type in the terminal:
```
python mobileposer/data_visualizer.py
```

When the terminal says: "Server start. Waiting for unity3d to connect." Click play on the 
"Figure_hand"
scene. Now, click the "Load File" button to load the **ground truth** file that you are trying to visualize. If you are trying to visualize a prediction, you still need to load this ground truth. If you are not trying to visualize a prediction, you can simply click into the "scene" tab in Unity so you can navigate the camera, and the blue hand should display the ground truth of the data that you loaded. The red hand displays the prediction, so if you are not trying to visualize a prediction then you can ignore the red hand. If you are trying to visualize a prediction, click the "Load Predictions" button and the red hand should display the predicted hand pose. Right now, predicted hand position has not been implemented. 



# Important Files

- data_recorder.py is used to take data (requires a long setup with specific hardware)
- pre_process.py takes the raw data in /raw_data and turns it into /processed_data
- data_visualizer.py helps visualize the processed data *most important*. 
If you set the unityp variable in main to false, then you can just look at EIT data. Default is true. 
To use: run data_visualizer.py. When it says "waiting for unity3d", play the Figure_hand scene in unity. Once the GUI
pops up, load a data file (with or without predictions). The hand should start moving in unity once you click play.
- get_hand_pos.py is used for forward kinematics. To use, run it and play the FK_hand scene in unity. Type in the name of the data file you wish to add hand positions to. 





