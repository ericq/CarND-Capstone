Team lead: Eric Qian
Team members: Ziwei Zeng, Yisi Liu

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

* Speical note about Tensorflow-Gpu
  * this program is ok to use Tensorflow-cpu. the inference speed is good enough (<90ms) in a CPU mode. In case you want to use Tensorflow-GPU, please make sure the version is latest Tensorflow-GPU as per the requirement.txt. And the CUDA 9.0 and CUDNN 7.4.24 (supported version for Ubuntu 16.04)
  * to simplify your setup and validation process, simply use the Docker Installation below, which is the main way the app is developed and tested.

### Docker Installation

Clone this project repository assuming you're on Mac or Linux developerment environment
# go to places where you want to place your project folder
```
git clone https://github.com/ericq/CarND-Capstone.git
```

[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file.
Note that this will start the docker container and mount host $PWD (project folder) directory to the 
docker container as '/capstone' directory; the same for log file direcotry. 
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --name my_capstone --rm -it capstone
```

Open additional temrinal to access the docker container.
```bash
docker exec -it my_capstone bash
```

### Usage
1. Go to the project folder
```bash
cd /capstone
```

2. Install python dependencies
```bash
apt-get install vim
pip install -r requirements.txt
# in above operation, please ensure that the tensorflow, numpy, pillow all are updated per the specified version in the requirements.txt

```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
