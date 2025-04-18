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
# Running Files

To run files, run them from the EITSync directory like
```
python mobileposer/[nameoffile].py
```


# Important Files

- data_recorder.py is used to take data (requires a long setup with specific hardware)
- pre_process.py takes the raw data in /raw_data and turns it into /processed_data
- data_visualizer.py helps visualize the processed data *most important*. 
If you set the unityp variable in main to false, then you can just look at EIT data. Default is true. 
To use: run data_visualizer.py. When it says "waiting for unity3d", play the Figure_hand scene in unity. Once the GUI
pops up, load a data file (with or without predictions). The hand should start moving in unity once you click play.
- get_hand_pos.py is used for forward kinematics. To use, run it and play the FK_hand scene in unity. Type in the name of the data file you wish to add hand positions to. 





