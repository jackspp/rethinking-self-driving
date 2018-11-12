"""
This script contains utils for self driving.
"""

from termcolor import *
import numpy as np
from PIL import Image
import keras
import numpy as np
import random
import time
import sys
import argparse
import logging
import os
import keyboard
import h5py as h5py
from keras.applications.inception_v3 import preprocess_input
import pandas as pd
from keras.models import Model
from IPython.core.debugger import Tracer
import scipy
from keras.utils import multi_gpu_model
from keras.layers import BatchNormalization, Concatenate, \
         LeakyReLU, Input, SeparableConv2D,  \
         UpSampling2D, Add, Conv2D, ZeroPadding2D,\
         Activation, Cropping2D, Lambda
from keras.layers.core import Flatten, Dense, Activation, Reshape, Dropout
from keras.layers.convolutional import  MaxPooling2D
from carla.agent import Agent
import importlib
from scipy import misc
import keras.backend as K

# Add extra python path
sys.path.insert(1, "/home/li/carla-0.8.2/PythonClient/bran_in_bran")
from utils.utils import PsudoSequential
from utils.predict_view_save import predict_view_save


#Set Gpu option for keras.
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))



sys.path.append("./model_utils")

class ModelControl(Agent):

    def __init__(self, args):

        self.img_size = args.rgb_shape[:2]
        self.model = None
        self.test_benchmark = args.test_benchmark
        self.predict_view_save = args.predict_view_save
        self.input_speed = args.input_speed
        self.save_input = args.save_input

        if self.test_benchmark:
            direction_list = ["REACH_GOAL",
                              "NONE",
                              "LANE_FOLLOW",
                              "TURN_LEFT",
                              "TURN_RIGHT",
                              "GO_STRAIGHT"]
            self.dir_list = direction_list

        ### ADD a new list for storing time delay info! ######
        self.td_list = [0,] * 11

        self.step = 0


    def load_model(self, args):
        """
        Build Model and load weights.

        #Params:
        -args:
            args.model_path, args.model_weights

        #Return: model

        """
        # Import net dynamically
        dir_path, net_name = os.path.split(args.net)

        sys.path.append(dir_path)

        net_name = net_name.split(".py")[0]
        net_module = importlib.import_module(net_name)
        net_class = getattr(net_module, "BranInBranNet")

        base_model = self.build_model(args, net_class)
        base_model.load_weights(args.model_weights)

        model = base_model
        #model = Model(base_model.input, base_model.outputs[-1])

        self.model = model

    def camera2model_input(self, sensor_data, image_size):

        if self.test_benchmark == False:
            camera_img = sensor_data["MyCamera"].data
        else:
            camera_img = sensor_data["CameraRGB"].data

        model_img = camera_img
        #model_img = scipy.misc.imresize(camera_img, image_size)
        model_img = model_img.astype(np.float32)
        #Preprocess
        image_input = np.expand_dims(model_img, axis=0)


        return image_input / 255.0


    def run_step(self, measurements, sensor_data, directions, target):

        if not self.model:
            raise Exception("Model not loaded,\
                            please load model first!")

        # Preprocess captured img,
        # Turn it into model input size, and preprocess it.
        batch_img = self.camera2model_input(sensor_data, self.img_size)

        target_img = np.squeeze(batch_img)
        # Controlled by AI model
        if self.input_speed == True:
            ############### TIME DELAY 5 FRRAMES !!! TOTOAL=2*5 +1 = 11

            # Current speed store
            current_speed = measurements.player_measurements.forward_speed
            current_speed = current_speed * 3.6/ 83.0
            self.td_list[0] = current_speed

            in_td_list = np.array([self.td_list])
            outputs = self.model.predict([batch_img, in_td_list], batch_size=1,
                                         verbose=0)


        else:
            outputs = self.model.predict(batch_img, batch_size=1, verbose=0)

        # Preprocess captured img,
        # Turn it into model input size, and preprocess it.

        if self.predict_view_save==True:

            predict_view_save(target_img, outputs, self.step)

        control = outputs[3]

        control = control.reshape(-1)
        print self.dir_list[int(directions)]

        # Extract corresponding control info from the outputs
        if directions == 0 or directions == 2:
            steer = control[0]
            if control[1] > 0:
                throttle = control[1]
                brake = 0
            else:
                throttle = 0
                brake = -control[1]

        elif directions == 3:
            steer = control[2]
            if control[3] > 0:
                throttle = control[3]
                brake = 0
            else:
                throttle = 0
                brake = -control[3]
        elif directions == 5:
            steer = control[4]
            if control[5] > 0:
                throttle = control[5]
                brake = 0
            else:
                throttle = 0
                brake = -control[5]
        elif directions == 4:
            steer = control[6]
            if control[7] > 0:
                throttle = control[7]
                brake = 0
            else:
                throttle = 0
                brake = -control[7]

        speed = control[8] * 83

        print("Output contrl info:%s"%str([steer,throttle,brake,speed]))
        print("Time Delay Info:%s"%(np.squeeze(self.td_list)))

        if brake > 0:
            brake = 1

        if measurements.player_measurements.forward_speed / 0.28 >= 35:
            brake = 1
            throttle = 0
        ############ MODIFIED TD_LIST ######
        if self.input_speed == True:
            self.td_list.pop(-1)
            self.td_list.pop(-1)
            #Tracer()()
            self.td_list.insert(1, current_speed)
            self.td_list.insert(1, steer)



        control = measurements.player_measurements.autopilot_control
        control.steer = steer

        if self.step < 50:
            control.throttle = 0.5
            control.brake = 0
        else:
            control.throttle = throttle
            control.brake = brake

        self.step += 1

        #save rgb img
        if self.save_input == True:
            sensor_data["CameraRGB"].save_to_disk("_model_test/%i"%self.step)


        return control


    def build_model(self, args, net_class):

        # Build Model

        bran_in_bran_net = net_class(args.rgb_shape,
                                        args.depth_shape,
                                        args.classes,
                                        args.strides,
                                        args.w_pad)

        return bran_in_bran_net.model





























