# Autonomous Systems Simulation

#### Table of Contents
- [Introduction](#introduction)
- [Features](#features)
  - [Object Tracking](#object-tracking)
  - [Bird's-eye View Mapping](#birds-eye-view-mapping)
  - [Line Detection](#line-detection)
- [Installation](#installation)
- [Usage](#usage)
- [Future Plans](#future-plans)
- [Improved Bird's-eye View](#improved-birds-eye-view)
  - [Next Path Prediction](#next-path-prediction)
  - [Navigator](#navigator)
  - [Collision Avoidance System](#collision-avoidance-system)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
This project simulates autonomous car systems using a game environment called City Car Gaming, known for its realistic mechanics and graphics, sufficient to track objects and simulate real-world scenarios.

## Features

#### Object Tracking
<img src="media\track0.png" alt="object_2" width="400"/><img src="media\track1.png" alt="object_2" width="400"/>
The system tracks various objects in the game environment using [yolov8n](weights) model.

### Bird's-eye View Mapping
<img src="media\bev0.png" alt="object_2" width="400"/><img src="media\bev1.png" alt="object_2" width="400"/>
Transforms the front camera view into a bird's-eye view to map surroundings and predict the next path of objects. Currently, it is able to:
- Show a bird's-eye view for close distance objects
- Warn about possible collisions based on short-range predictions but only in straight road for now

### Line Detection
<img src="media\line0.png" alt="object_2" width="400"/><img src="media\line1.png" alt="object_2" width="400"/>
Detects road lines using a sliding window technique, which assists in maintaining lane integrity.

## Installation
```
git clone https://github.com/elymsyr/autonomous-systems-simulation.git
pip install -r requirements.txt
python run_simulation.py
```
## Future Plans
### Improved Bird's-eye View
Enhancing the bird's-eye view system to generate a more detailed environmental map and implement car pose detection. The goal is to provide accurate 3D terrain detection and car orientation.

### Improved Next Path Prediction
Improving object path prediction to handle long-range scenarios and diverse object movement. The goal is to develop more accurate trajectory predictions for moving objects.

### Navigator
This system will provide self-driving capabilities by creating a path for the car to follow autonomously. The plan includes path planning algorithms and integration with the object detection module.

### Collision Avoidance System
Introducing a system to prevent collisions using real-time object detection and prediction. Future implementations might include autobraking or steering interventions.

## Contributing
Contributions are welcome. Please check the issues tab and submit pull requests. See .github

## License
See the [LICENSE](LICENSE) file for details.

## Acknowledgments

I would like to thank the following resources and projects that inspired or contributed to features:

- [PanopticBEV](https://panoptic-bev.cs.uni-freiburg.de/):
```
    @article{gosala21bev,
    author={Gosala, Nikhil and Valada, Abhinav},
    journal={IEEE Robotics and Automation Letters}, 
    title={Bird’s-Eye-View Panoptic Segmentation Using Monocular Frontal View Images}, 
    year={2022},
    volume={7},
    number={2},
    pages={1968-1975},
    doi={10.1109/LRA.2022.3142418}}
```
- [Rezaei, Mahdi & Terauchi, Mutsuhiro & Klette, Reinhard. (2015). Robust Vehicle Detection and Distance Estimation Under Challenging Lighting Conditions. IEEE Transactions on Intelligent Transportation Systems. ](https://www.researchgate.net/publication/274074437_Robust_Vehicle_Detection_and_Distance_Estimation_Under_Challenging_Lighting_Conditions)
- [The Ultimate Guide to Real-Time Lane Detection Using OpenCV](https://automaticaddison.com/the-ultimate-guide-to-real-time-lane-detection-using-opencv/)
