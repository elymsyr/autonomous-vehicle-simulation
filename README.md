# Autonomous Systems Simulation

#### Table of Contents
- [Introduction](#introduction)
- [Features](#features)
 - [Object Tracking](#object-tracking)
 - [Bird's-eye View Mapping](#bird's-eye-view-mapping)
 - [Line Detection](#line-detection)
- [Installation](#installation)
- [Usage](#usage)
- [Future Plans](#future-plans)
- [Improved Bird's-eye View](#improved-bird's-eye-view)
 - [Next Path Prediction](#next-path-prediction)
 - [Navigator](#navigator)
 - [Collision Avoidance System](#collision-avoidance-system)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project simulates autonomous car systems using a game environment called City Car Gaming, known for its realistic mechanics and graphics, sufficient to track objects and simulate real-world scenarios.

## Features

#### Object Tracking
The system tracks various objects in the game environment using [yolov8n](weights) model.
<img src="media\track0.png" alt="object_2" width="400"/><img src="media\track1.png" alt="object_2" width="400"/>

### Bird's-eye View Mapping
Transforms the front camera view into a bird's-eye view to map surroundings and predict the next path of objects. Currently, it is able to:
- Show a bird's-eye view for close distance objects
- Warn about possible collisions based on short-range predictions but only in straight road for now
<img src="media\bev0.png" alt="object_2" width="400"/><img src="media\bev1.png" alt="object_2" width="400"/>

### Line Detection
Detects road lines using a sliding window technique, which assists in maintaining lane integrity.
<img src="media\line0.png" alt="object_2" width="400"/><img src="media\line1.png" alt="object_2" width="400"/>

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
Contributions are welcome. Please check the issues tab and submit pull requests.

## License
See the [LICENSE](LICENSE) file for details.



I have a project and I need a detailed readme template includes a table of contents section.
My project aims to simulate autonomous car systems. Uses a car game called City Car Gaming with realistic mechanics and graphics enough to track objects. The current features are as follow:

1- Object Tracking: Cars, pedestrians, traffic warnings and everything
2- Bird's-eye View Mapping: Uses perpectives to transform front cam view to bird's-eye view. Also predicts the objects next path to warn any possible collision. It is still in early stages but bird's-eye view and path prediction works for close distance objects but needs to be improved.
3- Line Detection: It detects the road lines in front of the car using sliding window techniques.

There is also some future plans to be developed as such as,
1- Improved bird's-eye view that shows a detailed plan of the environment and pose detection for cars (and maybe training a model to detect terrarian).
2- Improved next path prediction for moving objects.
3- A system for creating a path for the to follow with self-driving (### Name the system you chatgpt)
4- A system for avoiding collisions. (Autobrake or else...)