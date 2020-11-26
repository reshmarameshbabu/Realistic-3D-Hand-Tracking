# Realistic-3D-Hand-Tracking

Depth-based 3D hand trackers are expected to estimate highly accurate poses of the human hand given the image. One of the critical problems in tracking the hand pose is
the generation of realistic predictions. This project proposes a novel filter called the “Humanistic Filter” that accepts a hand pose from a pose-estimator and generates the closest possible pose within the anatomical bounds of the real
human hand. The proposed filter can be plugged into any hand-pose estimator to enhance its performance. The filter has been tested on two state-of-the-art trackers. The empirical observations reveal that our proposed filter improves
the output from the viewpoint of anatomical correctness and also allows a smooth trade-off with pose error. The filter achieves the lowest error over the state-of-the-art trackers
at 10% correction.
