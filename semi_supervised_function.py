# -*- coding: utf-8 -*-

""" Deeplabv3+ model for Keras.
This model is based on TF repo:
https://github.com/tensorflow/models/tree/master/research/deeplab
On Pascal VOC, original model gets to 84.56% mIOU

Now this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers, but Theano will add
this layer soon.

MobileNetv2 backbone is based on this repo:
https://github.com/JonathanCMitchell/mobilenet_v2_keras

# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation,Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.python.keras.engine import get_source_inputs,Layer,InputSpec
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import get_file
from tensorflow.python.layers.utils import normalize_tuple,normalize_data_format

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"


NonLinearActivation = 'relu' #try selu as well (besides relu)

def run_on_cpu():
    num_cores = 4
    CPU=True
    GPU=False
    if GPU:
        num_GPU = 1
        num_CPU = 1
    if CPU:
        num_CPU = 1
        num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)
    
def return_iou(y_pred,y_val):
    t=0.5
    IOU_list=[]
    for j in range(y_pred.shape[0]):
        y_pred_ = np.array(y_pred[j,:,:] > t, dtype=bool)
        y_val_=np.array(y_val[j,:,:], dtype=bool)

        IOU = IoUOld(y_pred_, y_val_) 

        IOU_list.append(IOU)
    #now we take different threshholds, these threshholds 
    #basically determine if our IOU consitutes as a "true positiv"
    #or not 
    prec_list=[]
    for IOU_t in np.arange(0.5, 1.0, 0.05):
        #get true positives, aka all examples where the IOU is larger than the threshhold
        TP=np.sum(np.asarray(IOU_list)>IOU_t)
        #calculate the current precision, by devididing by the total number of examples ( pretty sure this is correct :D)
        #they where writing the denominator as TP+FP+FN but that doesnt really make sens becasue there are no False postivies i think
        Prec=TP/len(IOU_list)
        prec_list.append(Prec)

    return np.mean(prec_list)

class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(),training_data=() ,interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        #also load training data
        self.X_Val_T,self.Y_Val_T =training_data
        self.score_list=[0]
        self.score_list_train=[0]

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            y_pred_t = self.model.predict(self.X_Val_T, verbose=1)
            #first we have to threshhold
            prec_score=return_iou(y_pred,self.y_val)
            prec_score_t=return_iou(y_pred_t,self.Y_Val_T)
            
            print('Validation score is {}.Train score is {} Best so far is {}'.format(prec_score,prec_score_t, max(self.score_list)))
            
            if prec_score > max(self.score_list):
                print('Saving model-tgs-salt-IV.h5')
                self.model.save_weights('model-tgs-salt-IV.h5')
            
            self.score_list.append(prec_score)
            self.score_list_train.append(prec_score_t)
            
            
def _expand(tensor111, dim1, dim2):
    """
    Expands a tensor to have dimensions dim1 and dim2
    """
    ans = tf.keras.backend.repeat_elements(tensor111, dim1, 1)
    ans = tf.keras.backend.repeat_elements(ans, dim2, 2)
    return ans
    
    



def load_to_memory(tfrec=None,size=10000):


    #Argument the tf records file we want to load to memory. 
    valid_dataset = tf.data.TFRecordDataset(tfrec)
    valid_dataset = valid_dataset.map(parser)
    #kind of a trick to load the whole valiation dataset. 

    #TODO INCREASE 1000 or find better way

    valid_dataset = valid_dataset.batch(size)
    #this way we ge tis as a numpy array
    iterator = valid_dataset.make_one_shot_iterator()
    next_element=iterator.get_next()
    sess = tf.Session()
    dic,annotation=sess.run(next_element)
    sess.close()
    return dic,annotation

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_true=tf.to_float(y_true)
    smooth = 1.01
    intersection = tf.reduce_sum(tf.abs(y_true *y_pred), axis=-1)
    return (2. * intersection + smooth) / (tf.reduce_sum(tf.square(y_true),-1) + tf.reduce_sum(tf.square(y_pred),-1) + smooth)


def IoUOld(a,b):
    intersection = ((a==1) & (a==b)).sum()
    union = ((a==1) | (b==1)).sum()
    if union > 0:
        return intersection / union
    elif union == 0 and intersection == 0:
        return 1
    else:
        return 0
    




def BilinearUpsampling(size):
    return Lambda(lambda x: tf.image.resize_images(images=x,size=size))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(NonLinearActivation)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon,fused=False)(x)
    if depth_activation:
        x = Activation(NonLinearActivation)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon,fused=False)(x)
    if depth_activation:
        x = Activation(NonLinearActivation)(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN',fused=False)(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = int(inputs.shape[-1])
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN',fused=False)(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN',fused=False)(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN',fused=False)(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x

def deeplab_forward(img_input,input_shape=(128,128,3), classes=21, backbone='mobilenetv2', OS=16, alpha=1.,nummer=1,drop=1):
  

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = Conv2D(32, (3, 3), strides=(2, 2),
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN',fused=False)(x)
        x = Activation(NonLinearActivation)(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN',fused=False)(x)
        x = Activation(NonLinearActivation)(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)
        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                   skip_connection_type='conv', stride=2,
                                   depth_activation=False, return_skip=True)

        x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False)
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                depth_activation=False)

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                            skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                            depth_activation=False)
        x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                            skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                            depth_activation=True)

    else:
        OS = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same',
                   use_bias=False, name='Conv')(img_input)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='Conv_BN',fused=False)(x)
        x = Activation(relu6, name='Conv_Relu6')(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3, skip_connection=False)
        
        
        
        
        
        
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False)
        
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=10, skip_connection=False)
        
        
        
        
        
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=16, skip_connection=False)

    deep_rep=x

    # Image Feature branch
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5,fused=False)(b4)
    b4 = Activation(NonLinearActivation)(b4)
    #b4=Lambda(lambda x: (tf.image.resize_bilinear(x,(16,16))))(b4)
    b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)
    #b4=Lambda(lambda x: (tf.to_double(x)))(b4)
    #b4=tf.keras.layers.UpSampling2D((16,16))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5,fused=False)(b0)
    b0 = Activation(NonLinearActivation, name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5,fused=False)(x)
    x = Activation(NonLinearActivation)(x)
    if drop==1:
        x = Dropout(0.05,  name="drop_0")(x)
    if drop==0:
        x = Dropout(0.05,  name="drop_0")(x)

    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        #x = BilinearUpsampling((int(np.ceil(input_shape[0] / 4)),
        #                                    int(np.ceil(input_shape[1] / 4))))(x)
        x=tf.keras.layers.UpSampling2D((4,4))(x)
        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5,fused=False)(dec_skip1)
        dec_skip1 = Activation(NonLinearActivation)(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if classes == 21:
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x_low = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    #x=tf.keras.layers.UpSampling2D((4,4))(x)
    x = BilinearUpsampling((input_shape[0], input_shape[1]))(x_low)
    return x,x_low,deep_rep


def parser(record):
    '''Function to parse a TFRecords example'''
    
    features = tf.parse_single_example(
    record,
    features={
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string),
    'mask_raw': tf.FixedLenFeature([], tf.string),
    'depth': tf.FixedLenFeature([], tf.float32),
    'is_train': tf.FixedLenFeature([], tf.float32),
    })

    image = tf.decode_raw(features['image_raw'], tf.float64)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

    #height = tf.cast(features['height'], tf.int32)
    #width = tf.cast(features['width'], tf.int32)

    image = tf.reshape(image, (101, 101, 1))
    annotation = tf.reshape(annotation, (101, 101, 1))
    #,features['is_train']
    depth=tf.reshape(features['depth'],shape=(1,1,1))
    scaling=tf.reshape(features['is_train'],shape=(1,1))
    #scaling=features['is_train']
    #for prediction purposes it doesnt matter we jsut return two times the same here without droppout or anything. 
    return {'image_corrupt_1': image,'image_corrupt_2': image, 'depth': depth,'scaling':scaling},annotation

#parser with many different modfications
def parser_train(record):
    '''Function to parse a TFRecords example'''
    
    features = tf.parse_single_example(
    record,
    features={
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string),
    'mask_raw': tf.FixedLenFeature([], tf.string),
    'depth': tf.FixedLenFeature([], tf.float32),
    'is_train': tf.FixedLenFeature([], tf.float32),
    })

    image = tf.decode_raw(features['image_raw'], tf.float64)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

    #height = tf.cast(features['height'], tf.int32)
    #width = tf.cast(features['width'], tf.int32)

    annotation=tf.to_double(annotation)
    annotation = tf.reshape(annotation, (101, 101, 1))
    #,features['is_train']
    depth=tf.reshape(features['depth'],shape=(1,1,1))
    scaling=tf.reshape(features['is_train'],shape=(1,1))
    
    #we can add all sorts of stuff to the image
    image = tf.reshape(image, (101, 101,1))
    
    
    image_=tf.concat(axis=2,values=[image,annotation])
    
    #same flips for image and annotaition
    image_=tf.image.random_flip_left_right(image_)
    image_corrupt=image_[:,:,0]
    annotation=image_[:,:,1]
    annotation=tf.cast(annotation,dtype=tf.int8)
    annotation=tf.reshape(annotation, (101, 101, 1))
    image_corrupt=tf.reshape(image_corrupt, (101, 101, 1))
    
    #We make two corrupted versions of the Image. 
    image_corrupt_1=tf.image.random_brightness(image_corrupt,max_delta=0.1)
    #image_corrupt_1=tf.image.random_contrast(image_corrupt_1,lower=0,upper=1)
    image_corrupt_1=tf.image.random_jpeg_quality(image_corrupt_1,min_jpeg_quality=90,max_jpeg_quality=100)
    image_corrupt_1 =tf.clip_by_value(image_corrupt_1, 0.0, 1.0)
    image_corrupt_1=tf.layers.dropout(image_corrupt_1,rate=0.05,training=True)
    #image_corrupt_1=tf.image.grayscale_to_rgb(image_corrupt_1)
    
    image_corrupt_2=tf.image.random_brightness(image_corrupt,max_delta=0.1)
    #image_corrupt_2=tf.image.random_contrast(image_corrupt_2,lower=0,upper=1)
    image_corrupt_2=tf.image.random_jpeg_quality(image_corrupt_2,min_jpeg_quality=90,max_jpeg_quality=100)
    image_corrupt_2 = tf.clip_by_value(image_corrupt_2, 0.0, 1.0)
    image_corrupt_2=tf.layers.dropout(image_corrupt_2,rate=0.05,training=True)
    #image_corrupt_2=tf.image.grayscale_to_rgb(image_corrupt_2)
    
    #mage = tf.reshape(image, (101, 101, 1))
    return {'image_corrupt_1': image_corrupt_1,'image_corrupt_2': image_corrupt_2, 'depth': depth,'scaling':scaling},annotation


