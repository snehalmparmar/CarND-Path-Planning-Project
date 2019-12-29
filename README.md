# CarND-Path-Planning-Project

> Path Planning Project for Self-Driving Car ND

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![Path Planning](https://user-images.githubusercontent.com/4352286/44061856-0c255b1c-9f28-11e8-9b5a-0857b05d6da3.png)

![PathPlanningOutputVideo](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/video_output/2019-12-28-21-33-56.gif)

The goal of this project is to design a path planner that is able to create smooth, safe paths for the car to follow along a 3 lanes highway with traffic. A successful path planner will be able to keep inside its lane, avoid hitting other cars, and pass slower moving traffic all by using localization, sensor fusion, and map data.

In other words, the car :

- **Should be able to drive at least 4.32 miles without incident**. Incidents include exceeding acceleration/jerk/speed, collision, and driving outside of the lanes.
- **Should be able to drive according to the speed limit (50 mph)**. The car shouldn't drive faster than the speed limit. Also the car shouldn't drive much slower than speed limit unless obstructed by traffic.
- **Does not exceed max Acceleration (10 m/s^2) and Jerk (10 m/s^3)**.
- **Must not come into contact with any of the other cars on the road**.
- **Stays in its lane**, except for the time between changing lanes. The car doesn't spend more than a 3 seconds length outside the lanes during changing lanes, and every other time the car stays inside one of the 3 lanes on the right hand side of the road.
- **Should be able to smoothly change lanes** when it makes sense to do so, such as when behind a slower moving car and an adjacent lane is clear of other traffic.

## Overview
Starting to work on this project consists of the following steps:

1. Clone this repository
2. Build & compile the main program 
    - `mkdir build && cd build`
    - `cmake .. && make`
3. Launch `./path_planning`
4. Launch the Udacity Term 3 simulator
5. Enjoy!

---

## Installation and Dependencies

This project involves the Udacity Term 3 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).

### Other Important Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    
Once all the dependencies have been installed **clone** the project:

```sh
git clone https://github.com/snehalmparmar/CarND-Path-Planning-Project.git
```

and follow the steps 2 to 5 of the [Overview section](#overview) in order to build and run the main program.

---

## Implementation

My implementation is composed of two main parts which are the [prediction and behavior planning](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/main.cpp#L404), and the [trajectory generation](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/main.cppL407). See details below. 

### 1. Prediction/Behavior

The prediction and behavior planning has been implemented under the `SimpleBehaviorPlanner` function starting [from line 179](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/main.cpp#179). Based on the current state of the ego car and data from sensor fusion, the system decides what the car should do. By interpreting and understanding the surrounding environment, it finds the best possible behavior to adopt in order to drive safely and efficiently:

- We first look at cars in front of us blocking the traffic ([see lines 188 to 224](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/main.cpp#L188)). If it is the case, the car must consider to change lane ([see lines 224 to 227](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/main.cpp#L224)). 
- Otherwise, the car stays in the same lane and keep going by speeding up until we reach the speed limit (see [lines 228 to 231](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/main.cpp#L228)).

Lane changing is handle by a distinct function named `TryChangingLane` [starting from line 152](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/main.cpp#L152). This function is checking lanes for safety based on distance from cars and current speed [see line 55 to 78](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/main.cpp#L55). If no other lane is safe to drive, the ego car stays on its lane and slow down to reach the front car speed. 

### 2. Trajectory

Thanks to the previous step, we know how the car should behave. Whether it's has to change lane or drive straight, its path must be as smooth as possible. This is done under the `GenerateTrajectory` function starting [from line 234](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/main.cpp#L234). 

Trajectory calculation is initiated using the [spline library](http://kluge.in-chemnitz.de/opensource/spline/). To ease computation the map coordinates are transformed to local car coordinates ([see lines 276 to 281](https://github.com/snehalmparmar/CarND-Path-Planning-Project/blob/master/src/main.cpp#L276)). Points are then sampled from spline result and added to previous remaining path points in order to ensure more continuity to the trajectory.

---

## Possible Improvements

- Implement emergency braking
- Enhance trajectory generation by:
    1. Sampling a large number of end configurations near the desired position where we want the car to drive
    2. Generate corresponding trajectories and discard all non-drivable ones
    3. Pick the best trajectory based on jerk-minimizing cost function

---

## Questions or Feedback

> Contact me for any questions regarding this project.

* Snehal Parmar
* Email: [snehalparmar.nvs@gmail.com](mailto:snehalparmar.nvs@gmail.com)