"""
B Forys, brandon.forys@alumni.ubc.ca

Config file for applying a model to a streaming video.
Edit the variables in this file, then run `python AnalyzeVideos_stream.py` in
Evaluation-Tools.
"""

# Analysis Network parameters:
scorer = 'Rene'
Task = 'side_right_paw_movement'
date = 'June12_18'
trainingsFraction = 0.95  # Fraction of labeled images used for training
resnet = 50
snapshotindex = -1
shuffle = 1

# IP addresses of PC and Raspberry Pi:
PC_IP = ''
PI_IP = ''

# For plotting:
trainingsiterations = 2000  # type the number listed in .pickle file
pcutoff = 0.1  # likelihood cutoff for body part in image
# delete individual (labeled) frames after making video?
deleteindividualframes = False
