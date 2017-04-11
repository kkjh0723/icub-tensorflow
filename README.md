# icub-tensorflow
Visuo-motor learning of simulated iCub robot using a deep learning model implemented in Tensorflow (tested on simple toy task)

# Requirements
* C++
* Yarp and iCub (iCub_SIM)
  * [iCub software installation from source](http://wiki.icub.org/wiki/Linux:Installation_from_sources)
  * NOTE: in order to manipulate cylinder objects in the simulator, cylinder-cylinder collision is required because the robot's finger is also a cylinder. You need to install ODE with `--enable-libccd` option. [intalling ODE](http://wiki.icub.org/wiki/Linux:_Installing_ODE) 
  * Screen inside the simulator needs to be ON (It is set to show just checker board image in this sample experiment, but we are going to use it for gesture recognition task later). In order to turn on the screen, you need to change `screen` to `on` in `iCub_parts_activation.ini` inside `{iCub-folder}/contexts/simConfig` directory.
  * In order to make the manipulation easier, it is better to change the friction between objects and hands. We set the friction coefficient very large (1000000.0). You can change `contactFrictionCoefficient` in `ode_params.ini` inside `{iCub-folder}/contexts/simConfig` directory.
  * You can find more information about iCub simulatior in [simulator README](http://wiki.icub.org/wiki/Simulator_README)
* Yarp python biding 
  * [Yarp python binding](http://wiki.icub.org/wiki/Python_bindings)
  * iCub-python binding might not be required to run this codes 
* Python 2.7
* Tensorflow
  * [Tensorflow installation](https://www.tensorflow.org/install/)

# Tested environment
* Ubuntu 14.04
* Yarp 2.3.66
* iCub 1.4.0
* Python 2.7 (Anaconda)
* Tensorflow 0.12.x

# Task & Model
* We tested the program in a very simple object manipulation task. There is only one object (which can be different shapes and sizes) in front of the robot and the robot learned to grasp the object. 
* [NOTE] Since the purpose of this code is to test connecting icub simulator and tensorflow, we ignore some necessary components for training and examining deep neural networks. First, we don't use any validation data although checking validation loss was implemented in the training code (we use same training data to validation). Second, we don't test generalization but only check whether the network do operate as it trained. The testing situation on the simulator is exactly same as the training data. We may conduct real research experiments based on this code in the future.      
* We trained a CNN-RNN structure especially VMDNN [1] (or CNN-LSTM) model for the testing  

# Organization of the code
* For training model
* For online testing in the simulator 

# Training
1) Run `train_rnn.py`
  * Some options
    * `data_dir` and `data_fn`: for specify dataset location and file name
    * `log_dir`: location for saving Tensorflow checkpoint file
    * `device`: select CPU or GPU for training
    * Some hyper parameters for the model (e.g. `lr`, `batch_size`,  and etc)
    

# Testing in the iCub simultor (iCub_SIM)
1) Launch the simulator
  * in one terminal: `yarpserver`
  * in another terminal: `iCub_SIM`
2) Build online testing program
  * move to `onlineTestingProgram` folder and then move to`build` folder (make if there isn't)
  * `cmake ../`
  * `make`
3) Run online testing program
  * Run `python main.py` in `onlineTestingProgram` folder (you might need to provide proper options e.g. `--log_dir ./../log_dir01_01')
    * Some options
       * `log dir`: a directory which contains check point file of trained model
       * `max_leng`: maximum time steps you want to test
       * `use_data_vision`,`use_data_motor`, and `use_data_vision` : for checking offline trained performance and debugging 
       * `save_states` and `save_dir`: for saving internal states of the model while testing.
  * Run `./worldManipulator`, `./fingerGrashper`, `./vision`, `./screenWriter` and then `./controller` in seperate terminals (`./controller` should be the last one)
    * [NOTE] you can make a simple script to run all programs in one terminal (refer to `llauncher` and `killer` in `onlineTestingProgram` folder)  
 

# References
