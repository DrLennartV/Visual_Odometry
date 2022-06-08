### Monocular Visual Odometry

    The challenge is to track the path of the camera using information from the videos captured. The algorithm is described here https://en.wikipedia.org/wiki/Visual_odometry


- A skeleton code is provided in `Odometry.py` , with necessaary structure for submitting to gradescope
- A sample video is given to test your methods and an evaluation code to benchmark your approaches
- The video is processed for lens distortion and converted to frames. The frames are numbered in the order they appear in the video
- Your algorithm will be evaluated on Gradescope autograder using another video which is not provided here
- The camera parameters for both the videos are present in the file `calib.txt`
- The ground truth for the the sample video is present in `gt_sequence.txt`. The process of evalutation and computing the final error from predictions is outlined in the evaluation notebook.
