
"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu
Modified by B Forys, brandon.forys@alumni.ubc.ca

This script analyzes a streaming video based on a trained network.
You need tensorflow for evaluation. Run by:
CUDA_VISIBLE_DEVICES=0 python3 AnalyzeVideos_stream.py

"""

####################################################
# Dependencies
####################################################

import os.path
import sys

subfolder = os.getcwd().split('Evaluation-Tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "pose-tensorflow/")
sys.path.append(subfolder + "Generating_a_Training_Set")

from myconfig_stream import Task, date, \
    trainingsFraction, resnet, snapshotindex, shuffle

# Deep-cut dependencies
from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input

import pickle
# import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
import skimage
import skimage.color
import pandas as pd
import numpy as np
import os
import io
# import tkinter
import socket
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import time
import datetime
import asyncio
# import multiprocessing as mp
import concurrent.futures
from PIL import Image, ImageTk

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def getpose(image, cfg, outputs, outall=False):
    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose


####################################################
# Loading data, and defining model folder
####################################################

basefolder = '../pose-tensorflow/models/'  # for cfg file & ckpt!
modelfolder = (basefolder + Task + str(date) + '-trainset' +
               str(int(trainingsFraction * 100)) + 'shuffle' + str(shuffle))
cfg = load_config(modelfolder + '/test/' + "pose_cfg.yaml")

##################################################
# Load and setup CNN part detector
##################################################

# Check which snap shots are available and sort them by # iterations
Snapshots = np.array([
    fn.split('.')[0]
    for fn in os.listdir(modelfolder + '/train/')
    if "index" in fn
])
increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]

print(modelfolder)
print(Snapshots)

##################################################
# Compute predictions over images
##################################################

# Check if data already was generated:
cfg['init_weights'] = modelfolder + '/train/' + Snapshots[snapshotindex]

# Name for scorer:
trainingsiterations = (cfg['init_weights'].split('/')[-1]).split('-')[-1]

# Name for scorer:
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(
    date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)
cfg['init_weights'] = modelfolder + '/train/' + Snapshots[snapshotindex]
sess, inputs, outputs = predict.setup_pose_prediction(cfg)
pdindex = pd.MultiIndex.from_product(
    [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
    names=['scorer', 'bodyparts', 'coords'])

##################################################
# Datafolder
##################################################

start = time.time()
PredicteData = np.zeros((1000, 3 * len(cfg['all_joints_names'])))
index = 0
x_range = list(range(0,(3 * len(cfg['all_joints_names'])),3))
y_range = list(range(1,(3 * len(cfg['all_joints_names'])),3))
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

x_arr = []
y_arr = []
x_overall = []
y_overall = []
threshold = 0

# Initialize IP addresses
PC_IP = '169.254.205.9'
PI_IP = '169.254.169.188'

# Set up async loop
async def frame_process(image, cfg, outputs, index):
    pose = getpose(image, cfg, outputs)
    print('Pose ' + str(index) + ' saved!')
    PredicteData[index, :] = pose.flatten()

# Initiates UDP communication (PC -> Pi)
def udp_handler(threshold):
    try:
        udp_socket.sendto(bytes(str(threshold), "UTF-8"), \
        (PI_IP, 5007))
        print("sending request")
    except:
        print("Could not send request.")

loop = asyncio.new_event_loop()

# Sets up UDP connection (PC -> Pi)
try:
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('', 5007))
except socket.error:
    print("Failed to bind to socket")

# Start TCP connection (Pi -> PC)
server_socket = socket.socket()
server_socket.bind((PC_IP, 8000))
server_socket.listen(0)

print("Starting to extract posture")
# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')
try:
    while True:
        # with concurrent.futures.ProcessPoolExecutor() as executor:

        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_input = Image.open(image_stream)
        w, h = image_input.size
        image = (img_as_ubyte(image_input))
        # Predict movement data from frame
        task = loop.create_task(frame_process(image, cfg, outputs, index))
        loop.run_until_complete(task)

        # pose = getpose(image, cfg, outputs)
        # print('Pose ' + str(index) + ' saved!')
        # PredicteData[index, :] = pose.flatten()

        # Plot predicted movement data on frame
        # plt.axis('off')
        # plt.figure(frameon=False, figsize=(w * 1. / 100, h * 1. / 100))
        # plt.subplots_adjust(
        #     left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # plt.imshow(image_input)
        # for x_plt, y_plt, c in zip(x_range, y_range, colors):
        #     plt.scatter(
        #         PredicteData[index, :][x_plt],
        #         PredicteData[index, :][y_plt],
        #         color=c, alpha=0.2)
        #
        # plt.xlim(0, w)
        # plt.ylim(0, h)
        # plt.subplots_adjust(
        #     left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # plt.gca().invert_yaxis()
        # plt.show()

        for a in PredicteData[index, 0:14:3]:
            x_arr.append(a)
        for b in PredicteData[index, 1:15:3]:
            y_arr.append(b)

        x_avg = np.mean(x_arr)
        y_avg = np.mean(y_arr)
        x_overall.append(x_avg)
        y_overall.append(y_avg)
        x_stdev = np.std(x_overall)
        y_stdev = np.std(y_overall)
        print(x_stdev)

        if(abs(x_overall[index] - x_overall[index - 1]) >= x_stdev and \
        abs(y_overall[index] - y_overall[index - 1]) >= y_stdev and \
        index >= 1):
            threshold = 1
        print(str(threshold))

        # Initiates UDP communication (PC -> Pi)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(udp_handler, threshold)

        # processes[1].join()

        x_arr = []
        y_arr = []

        # plt.savefig('frame{}.png'.format(str(index)))
        # frame = Image.open('frame{}.png'.format(str(index)))

        threshold = 0
        index += 1

except KeyboardInterrupt:
    print("Finished.")

finally:
    connection.close()
    server_socket.close()
    stop = time.time()
    stop_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    dictionary = {
        "start": start,
        "stop": stop,
        "run_duration": stop - start,
        "Scorer": scorer,
        "config file": cfg,
        "frame_dimensions": (w, h),
        "nframes": index
    }
    metadata = {'data': dictionary}

    print("Saving results...")
    PredicteData = PredicteData[0:index, :]
    DataMachine = pd.DataFrame(
        PredicteData, columns=pdindex, index=range(index))
    AvgsMachine = pd.DataFrame(
    {'x': x_overall, 'y': y_overall}
    )

    DataMachine.to_csv("Run" + str(stop_time) + '.csv')
    AvgsMachine.to_csv("Run_avgs_" + str(stop_time) + '.csv')

    print("Results saved!")
