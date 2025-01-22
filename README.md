# football_analysis
# Overview:
This project analyses football matches using advanced computer vision techniques, including:

- YOLO for object detection (players, ball, referees).
- ByteTrack for tracking movements.
- KMeans and SigLip for team classification.
# Key Components
1. Input and Output Processing
- OpenCV is utilized for:
Extracting frames from video inputs.
Applying visualizations on the processed frames.
Saving the processed frames back into video format.
2. Object Detection and Tracking
YOLOv11n is used for object detection.
ByteTrack is utilized for tracking players, ball, and referees.
3. Team Classification
Workflow:
- Player Cropping:

Player crops are extracted from video frames using YOLO detection boxes.
- Background and Torso Clustering:

KMeans is applied to separate the background from player torsos.
- Team Color Classification:
The colors of players' shirts are clustered to distinguish between two teams.
3. Ball Interpolation
Workflow:
Basic Idea

The Ball is not detected in many frames, but we know that in most cases, the ball travels in a linear fashion, and in cases where passes are curved, still linear interpolation can be used to fill in missing values.
Interpolation

We separate out the frames when there is no ball detection in our results dataframe, then insert np.nans into the box column for those and merge the dataframe.
Finally, we replace the nans by using linear interpolation
