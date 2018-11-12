"""
##### Note this net's input is only rgb image!!! ########
"""

import numpy as np
import tensorflow as tf
import keras
import os
import sys

import keras.backend as K
from keras.models import Model
from keras import callbacks, optimizers, initializers
from keras.utils import multi_gpu_model
from keras.layers import BatchNormalization, Concatenate, \
     LeakyReLU, Input, SeparableConv2D,  \
     UpSampling2D, Add, Conv2D, ZeroPadding2D,\
     Activation, Cropping2D, Lambda, SpatialDropout2D
from keras.layers.core import Flatten, Dense, Activation, Reshape, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from IPython.core.debugger import Tracer
from utils.utils import PsudoSequential, Softmax4D
#from utils.pixelShuffle import PixelShuffler



class BranInBranNet(object):
    """
    Use ideas of fuse-net for inferencing segmentation from depth and rgb images ,
    Dilation Convolution.
    """

    def __init__(self, rgb_shape, depth_shape, \
                 classes=13, img_width=200, img_height=88, \
                 strides=8, w_pad=2):

        self.in_rgb = Input(shape=rgb_shape,)
        #self.in_depth = Input(shape=depth_shape,)
        self.classes = classes
        self.img_width = rgb_shape[1]
        self.img_height = rgb_shape[0]
        self.strides = strides
        self.w_pad = w_pad

        self.build_net()



    def build_net(self):
        """
        Reuse the idea from fuse-net,
        RGB image + depth image ---> segmentation, depth image,

        #Encoder + Decoder Model
        """

        #### Modified! ######
        #### Merge branch control part with perceptron part! ########


        # Initialize
        self.dilate_perceptron_and_driving_module()


        self.inference()

        self.model = Model(inputs=[self.in_rgb], outputs=self.visual_outputs)

        return self.model


    def dilate_perceptron_and_driving_module(self):
        """
        Two Inputs, rgb images and depth map,
        Three Outputs, rgb images, depth map, segmentation result

        Important Notes:
            - Dilation Conv to enlarge receptive field
            - Dilation Rate 1,2,3 to solve "gridding" problem
            - Decoder: Dense Upsampling Convolution instead of deconv
              or binear interpolation
            - Separable with Linear BottleNeck from MobileNet2

        #### Dilated Convolution

        - [Multi-Scale Context Aggregation by Dilated Convolutions]
          (https://arxiv.org/abs/1511.07122)


        """

        ########## function define #####################3
        def res_block(name,
                d_chan_num,
                in_chan_num,
                activation="relu",
                atrous=False):

            prefix = "per_"
            seq = PsudoSequential()
            if atrous == False:
                seq.add(Conv2D(d_chan_num, (3,3), padding="same",name=prefix+name+"_res_conv1"))
                seq.add(BatchNormalization(name=prefix+name+"_res_bn1"))
                seq.add(Activation(activation, name=prefix+name+"_res_act1"))
                seq.add(Conv2D(in_chan_num, (3,3), padding="same", name=prefix+name+"_res_conv2"))
                seq.add(BatchNormalization(name=prefix+name+"_res_bn2"))

            else:
                seq.add(Conv2D(d_chan_num, (3,3), dilation_rate=(1,1),padding="same",
                        name=prefix+name+"_res_conv1"))
                seq.add(BatchNormalization(name=prefix+name+"_res_bn1"))
                seq.add(Activation(activation, name=prefix+name+"_res_act1"))
                seq.add(Conv2D(in_chan_num, (3,3), dilation_rate=(2,2), padding="same",
                            name=prefix+name+"_res_conv2"))
                seq.add(BatchNormalization(name=prefix+name+"_res_bn2"))
                seq.add(Activation(activation, name=prefix+name+"_res_act2"))
                seq.add(Conv2D(d_chan_num, (3,3), dilation_rate=(3,3),padding="same",
                            name=prefix+name+"_res_conv3"))
                seq.add(BatchNormalization(name=prefix+name+"_res_bn3"))

            return seq

        def res_add_relu(name,):
            prefix = "per_"
            seq = PsudoSequential()
            seq.add(keras.layers.Add(name=prefix+name+"_res_add"))
            seq.add(Activation("relu", name=prefix+name+"_res_add_relu"))
            return seq

        def incept_pool1(name, c_size):
            prefix = "per_"
            seq = PsudoSequential()
            seq.add(Conv2D(c_size, (2,2), strides=2, padding="valid", name=prefix+name))
            return seq

        def incept_pool2(name):
            prefix = "per_"
            seq = PsudoSequential()
            seq.add(MaxPooling2D(strides=2, name=prefix+name))
            return seq

        def incept_pool3(name):
            prefix = "per_"
            seq = PsudoSequential()
            seq.add(Concatenate(axis=-1, name=prefix+name))
            return seq

        def control_block1():
                prefix = "control_"
                seq = PsudoSequential()

                seq.add(BatchNormalization(name=prefix+"start_bn1"))
                seq.add(Conv2D(256,(1,1),
                    kernel_initializer=initializers.he_normal(),
                    name=prefix+"start_conv1"))
                seq.add(BatchNormalization(name=prefix+"start_bn2"))
                seq.add(Activation("relu", name=prefix+"start_act"))
                return seq

        def conv_activation(name,
                            activation=None,
                            shape=None):
            prefix = "control_"
            seq = PsudoSequential()
            seq.add(Conv2D(64, (3,3), padding="same", kernel_initializer=initializers.he_normal(),
                    name=prefix+name+"_conv1"))
            seq.add(SpatialDropout2D(0.3, name=prefix+name+"_drop1"))
            seq.add(BatchNormalization(name=prefix+name+"_bn1"))
            seq.add(Activation("relu",name=prefix+name+"_activation1"))

            seq.add(Conv2D(32, (3,3), padding="same", kernel_initializer=initializers.he_normal(),
                    name=prefix+name+"_conv2"))
            seq.add(SpatialDropout2D(0.3, name=prefix+name+"_drop2"))
            seq.add(BatchNormalization(name=prefix+name+"_bn2"))
            seq.add(Activation("relu",name=prefix+name+"_activation2"))

            seq.add(Conv2D(1, shape, kernel_initializer=initializers.he_normal(),
                    name=prefix+name+"_conv3"))
            seq.add(Activation(activation,name=prefix+name+"_activation3"))
            return seq




        ########## encoder first part!!!! #################3
        en_rgb_conv1 = PsudoSequential()
        en_rgb_conv1.add(Conv2D(32, (3, 3), padding="same",
                                #use_bias=False,
                                kernel_initializer=initializers.he_normal(),
                                name="rgb_conv1_1"))
        en_rgb_conv1.add(BatchNormalization(name="rgb_conv1_1_bn"))
        en_rgb_conv1.add(Activation("relu", name="rgb_conv1_1_act"))
        self.en_rgb_conv1 = en_rgb_conv1

        en_conv1 = PsudoSequential()
        en_conv1.add(Conv2D(32, (3, 3),  padding="same",
                            #use_bias=False,
                            kernel_initializer=initializers.he_normal(),
                            name="conv1_2"))
        en_conv1.add(BatchNormalization(name="rgb_conv1_2_bn"))
        en_conv1.add(Activation("relu", name="rgb_conv1_2_act"))
        self.en_conv1 = en_conv1

        ###########3 POOL1 #################
        self.incept_pool1_1 = incept_pool1(name="pool_conv1", c_size=32)
        self.incept_pool1_2 = incept_pool2(name="pool_max1")
        self.incept_pool1_3 = incept_pool3(name="pool_concat1")

        ######## res block1 ###########
        self.en_res1_1 = res_block("res1_1", 64, 64)
        self.en_res_add_1_1 = res_add_relu("res1_1",)

        self.en_res1_2 = res_block("res1_2", 64, 64)
        self.en_res_add_1_2 = res_add_relu("res1_2",)

        self.en_res1_3 = res_block("res1_3", 64, 64)
        self.en_res_add_1_3 = res_add_relu("res1_3")
            # Sptial_drop ########
        self.en_res1_sp = SpatialDropout2D(rate=0.2, name="res1_drop")

        ########## POOL2 ################
        self.incept_pool2_1 = incept_pool1(name="pool_conv2", c_size=64)
        self.incept_pool2_2 = incept_pool2(name="pool_max2")
        self.incept_pool2_3 = incept_pool3(name="pool_concat2")

        ############## res2 block ###########
        self.en_res2_1 = res_block("res2_1", 128, 128)
        self.en_res_add_2_1 = res_add_relu("res2_1",)

        self.en_res2_2 = res_block("res2_2", 128, 128)
        self.en_res_add_2_2 = res_add_relu("res2_2",)

        self.en_res2_3 = res_block("res2_3", 128, 128)
        self.en_res_add_2_3 = res_add_relu("res2_3")
            # Sptial_drop ########
        self.en_res2_sp = SpatialDropout2D(rate=0.2, name="res2_drop")

        ###########3 POOL3 #################
        self.incept_pool3_1 = incept_pool1(name="pool_conv3", c_size=128)
        self.incept_pool3_2 = incept_pool2(name="pool_max3")
        self.incept_pool3_3 = incept_pool3(name="pool_concat3")

        ########### res Atrous1 ###########################3
        self.en_res_atrous1 = res_block("res3_1", 256, 256)
        self.en_res_atrous1_add = res_add_relu("res3_1")

        ########## Spatial drop #################33
        self.en_res3_sp = SpatialDropout2D(rate=0.3, name="res3_drop")

        ########### res Atrous2 ###########################3
        self.en_res_atrous2 = res_block("res3_2", 256, 256)
        self.en_res_atrous2_add = res_add_relu("res3_2")

        ########## Spatial drop #################33
        self.en_res4_sp = SpatialDropout2D(rate=0.3, name="res4_drop")


        ###### Decoder part ##########
        de_main = PsudoSequential()
        de_main.add(Conv2D(512*4, (1, 1),
                            kernel_initializer=initializers.he_normal(),
                            name='final'))
        de_main.add(BatchNormalization(name="final_bn"))
        de_main.add(Activation("relu", name="final_act"))
        de_main.add(SpatialDropout2D(rate=0.5, name="final_drop"))
        self.de_main = de_main


        ########## Depth #####################
        de_depth = PsudoSequential()
        de_depth.add(Conv2D(256, (1,1), name="depth_conv1", padding="same",
                            kernel_initializer=initializers.he_normal(),))
        de_depth.add(BatchNormalization(name='depth_conv1_bn'))
        de_depth.add(Activation("relu", name="depth_act1"))
        de_depth.add(SpatialDropout2D(rate=0.3, name="depth_drop_1"))
        # Use upsampling
        de_depth.add(UpSampling2D((2,2)))
        de_depth.add(Conv2D(128, (3,3), name="depth_conv2", padding="same",
                            kernel_initializer=initializers.he_normal(),))
        de_depth.add(BatchNormalization(name='depth_conv2_bn'))
        de_depth.add(Activation("relu", name="depth_act2"))
        de_depth.add(SpatialDropout2D(rate=0.3, name="depth_drop_2"))

        de_depth.add(UpSampling2D((2,2)))
        de_depth.add(Conv2D(64, (3,3), name="depth_conv3", padding="same",
                            kernel_initializer=initializers.he_normal(),))
        de_depth.add(BatchNormalization(name='depth_conv3_bn'))
        de_depth.add(Activation("relu", name="depth_act3"))
        de_depth.add(SpatialDropout2D(rate=0.3, name="depth_drop_3"))

        de_depth.add(UpSampling2D((2,2)))
        de_depth.add(Conv2D(1, (3,3), activation="sigmoid", padding="same", name="depth"))
        self.de_depth = de_depth



        ############# Seg #####################
        de_seg = PsudoSequential()
        de_seg.add(Conv2D(256, (1,1), name="seg_conv1", padding="same",
                            kernel_initializer=initializers.he_normal(),))
        de_seg.add(BatchNormalization(name='seg_conv1_bn'))
        de_seg.add(Activation("relu", name="seg_act1"))
        de_seg.add(SpatialDropout2D(rate=0.3, name="seg_drop_1"))

        # Use upsampling
        de_seg.add(UpSampling2D((2,2)))
        de_seg.add(Conv2D(128, (3,3), name="seg_conv2", padding="same",
                            kernel_initializer=initializers.he_normal(),))
        de_seg.add(BatchNormalization(name='seg_conv2_bn'))
        de_seg.add(Activation("relu", name="seg_act2"))
        de_seg.add(SpatialDropout2D(rate=0.3, name="seg_drop_2"))


        de_seg.add(UpSampling2D((2,2)))
        de_seg.add(Conv2D(64, (3,3), name="seg_conv3", padding="same",
                            kernel_initializer=initializers.he_normal(),))
        de_seg.add(BatchNormalization(name='seg_conv3_bn'))
        de_seg.add(Activation("relu", name="seg_act3"))
        de_seg.add(SpatialDropout2D(rate=0.3, name="seg_drop_3"))

        de_seg.add(UpSampling2D((2,2)))
        de_seg.add(Conv2D(13, (3,3), activation="relu", padding="same",))
        de_seg.add(Softmax4D(name="seg"))
        self.de_seg = de_seg


            ###########  rgb #########################
        de_rgb = PsudoSequential()
        de_rgb.add(Conv2D(256, (1,1), name="rgb_conv1", padding="same",
                            kernel_initializer=initializers.he_normal(),))
        de_rgb.add(BatchNormalization(name='rgb_conv1_bn'))
        de_rgb.add(Activation("relu", name="rgb_act1"))
        de_rgb.add(SpatialDropout2D(rate=0.3, name="rgb_drop_1"))

        # Use upsampling
        de_rgb.add(UpSampling2D((2,2)))
        de_rgb.add(Conv2D(128, (3,3), name="rgb_conv2", padding="same",
                            kernel_initializer=initializers.he_normal(),))
        de_rgb.add(BatchNormalization(name='rgb_conv2_bn'))
        de_rgb.add(Activation("relu", name="rgb_act2"))
        de_rgb.add(SpatialDropout2D(rate=0.3, name="rgb_drop_2"))

        de_rgb.add(UpSampling2D((2,2)))
        de_rgb.add(Conv2D(64, (3,3), name="rgb_conv3", padding="same",
                            kernel_initializer=initializers.he_normal(),))
        de_rgb.add(BatchNormalization(name='rgb_conv3_bn'))
        de_rgb.add(Activation("relu", name="rgb_act3"))
        de_rgb.add(SpatialDropout2D(rate=0.3, name="rgb_drop_3"))

        de_rgb.add(UpSampling2D((2,2)))
        de_rgb.add(Conv2D(3, (3,3), activation="sigmoid", padding="same", name="rgb"))
        self.de_rgb = de_rgb

        ########### CONTROL MODULE! ###########
        self.controlblock1 = control_block1()

        ######## control_res block1 ###########
        self.en_c1_1 = res_block("c1_1", 256, 256)
        self.en_c_add_1_1 = res_add_relu("c1_1",)

        self.en_c_sp_add1 = SpatialDropout2D(rate=0.2, name="add_c1_drop")
        ######## control_res block2 ###########
        self.en_c1_2 = res_block("c1_2", 256, 256)
        self.en_c_add_1_2 = res_add_relu("c1_2",)

        self.en_c_sp_add2 = SpatialDropout2D(rate=0.2, name="add_c2_drop")


        ######## control_res block3 ###########
        self.en_c1_3 = res_block("c1_3", 256, 256)
        self.en_c_add_1_3 = res_add_relu("c1_3",)

        self.en_c_sp_add3 = SpatialDropout2D(rate=0.2, name="add_c3_drop")
        ######## control_res block4 ###########
        self.en_c1_4 = res_block("c1_4", 256, 256)
        self.en_c_add_1_4 = res_add_relu("c1_4",)

        ####### Sptial_drop ########
        self.en_c_sp1 = SpatialDropout2D(rate=0.2, name="c1_drop")

        ###########control POOL #################
        self.c_pool1_1 = incept_pool1(name="c_pool_conv1", c_size=256)
        self.c_pool1_2 = incept_pool2(name="c_pool_max1")
        self.c_pool1_3 = incept_pool3(name="c_pool_concat1")

        ######## control_res block5 ###########
        self.en_c2_1 = res_block("c2_1", 512, 512)
        self.en_c_add_2_1 = res_add_relu("c2_1",)

        ######## control_res block6 ###########
        self.en_c2_2 = res_block("c2_2", 512, 512)
        self.en_c_add_2_2 = res_add_relu("c2_2",)

        ######## control_res block7 ###########
        self.en_c2_3 = res_block("c2_3", 512, 512)
        self.en_c_add_2_3 = res_add_relu("c2_3",)

        ####### Sptial_drop ########
        self.en_c_sp2 = SpatialDropout2D(rate=0.3, name="c2_drop")

        shape = (self.img_height / (8*2), self.img_width / (8*2))

        ############ Final Control ###################
        self.f_steer = conv_activation("follow_steer",
                                        activation="tanh",
                                        shape=shape)
        self.l_steer = conv_activation("left_steer",
                                        activation="tanh",
                                        shape=shape)
        self.s_steer = conv_activation("straight_steer",
                                        activation="tanh",
                                        shape=shape)
        self.r_steer = conv_activation("right_steer",
                                        activation="tanh",
                                        shape=shape)
        self.f_gas = conv_activation("follow_gas",
                                        activation="tanh",
                                        shape=shape)
        self.l_gas = conv_activation("left_gas",
                                        activation="tanh",
                                        shape=shape)
        self.s_gas = conv_activation("straight_gas",
                                        activation="tanh",
                                        shape=shape)
        self.r_gas = conv_activation("right_gas",
                                        activation="tanh",
                                        shape=shape)
        self.speed = conv_activation("speed",
                                        activation="sigmoid",
                                        shape=shape)

        self.control_concat = Concatenate(axis=-1, name="control_concat")
        self.control_final = Lambda(lambda x: K.reshape(x, (-1, 9, 1)), name="control_outputs")


    def inference(self):
        """
        Three important notes:
            - in_rgb and in_depth share encoder part except conv1_1,
            - sum up with en_depth when inferencing en_rgb.
            - 3 decoders conresponds to 3 different groups of the final
              encoder layer's outputs.
        """

        ##############3 func define ###################
        def res_call(res_block, res_add, input_t,):
            inter_res = res_block(input_t)
            out_res = res_add([input_t, inter_res])
            return out_res


        def pool_call(pool1, pool2, pool3, input_t):
            pool1_out = pool1(input_t)
            pool2_out = pool2(input_t)
            pool3_out = pool3([pool1_out, pool2_out])
            return pool3_out


        ####### ENCODER #########
        rgb_1_1 = self.en_rgb_conv1(self.in_rgb)
        rgb_1_2 = self.en_conv1(rgb_1_1)
        pool1 = pool_call(self.incept_pool1_1, self.incept_pool1_2, self.incept_pool1_3, rgb_1_2)

        res1_1 = res_call(self.en_res1_1, self.en_res_add_1_1, pool1)
        res1_2 = res_call(self.en_res1_2, self.en_res_add_1_2, res1_1)
        res1_3 = res_call(self.en_res1_3, self.en_res_add_1_3, res1_2)
        res1_3 = self.en_res1_sp(res1_3)

        pool2 = pool_call(self.incept_pool2_1, self.incept_pool2_2, self.incept_pool2_3, res1_3)

        res2_1 = res_call(self.en_res2_1, self.en_res_add_2_1, pool2)
        res2_2 = res_call(self.en_res2_2, self.en_res_add_2_2, res2_1)
        res2_3 = res_call(self.en_res2_3, self.en_res_add_2_3, res2_2)
        res2_3 = self.en_res2_sp(res2_3)
        pool3 = pool_call(self.incept_pool3_1, self.incept_pool3_2, self.incept_pool3_3, res2_3)
        res_atr = res_call(self.en_res_atrous2, self.en_res_atrous2_add, pool3)
        res_atr = self.en_res4_sp(res_atr)


        ##### DECODER #######
        fusion_1 = self.de_main(res_atr)

        pred_depth = self.de_depth(fusion_1)
        pred_rgb = self.de_rgb(fusion_1)
        pred_seg = self.de_seg(fusion_1)

            ##### IINFERENCE FOR DIRECTION OUTPUT PART!!! ##############
        c_1 = self.controlblock1(res_atr)
        c_res_1 = res_call(self.en_c1_1, self.en_c_add_1_1, c_1)
        #c_res_1 =  self.en_c_sp_add1(c_res_1)
        c_res_2 = res_call(self.en_c1_2, self.en_c_add_1_2, c_res_1)
        #c_res_2 = self.en_c_sp_add2(c_res_2)
        c_res_3 = res_call(self.en_c1_3, self.en_c_add_1_3, c_res_2)
        #c_res_3 = self.en_c_sp_add3(c_res_3)
        c_res_4 = res_call(self.en_c1_4, self.en_c_add_1_3, c_res_3)

        c_res_4 = self.en_c_sp1(c_res_4)
        c_pool1 = pool_call(self.c_pool1_1, self.c_pool1_2, self.c_pool1_3, c_res_4)
        #c_res_5 = res_call(self.en_c2_1, self.en_c_add_2_1, c_pool1)
        #c_res_6 = res_call(self.en_c2_2, self.en_c_add_2_2, c_res_5)
        #c_res_7 = res_call(self.en_c2_3, self.en_c_add_2_3, c_res_6)
        #c_res_7 = self.en_c_sp2(c_res_7)

        #control_pool_out = c_res_7
        control_pool_out = c_pool1


        f_steer = self.f_steer(control_pool_out)
        l_steer = self.l_steer(control_pool_out)
        s_steer = self.s_steer(control_pool_out)
        r_steer = self.r_steer(control_pool_out)

        f_gas = self.f_gas(control_pool_out)
        l_gas = self.l_gas(control_pool_out)
        s_gas = self.s_gas(control_pool_out)
        r_gas = self.r_gas(control_pool_out)

        speed = self.speed(control_pool_out)

        control_conv2 = self.control_concat([f_steer, f_gas,
                                                l_steer, l_gas,
                                                s_steer, s_gas,
                                                r_steer, r_gas,
                                                speed])

        control_outputs = self.control_final(control_conv2)

        self.visual_outputs = [pred_depth, pred_rgb, pred_seg]

        self.visual_outputs.append(control_outputs)

# Test
if __name__ == "__main__":
    net = BranInBranNet((88,200,3), (88,200,1))
    #Tracer()()
    print("test done!")












































































