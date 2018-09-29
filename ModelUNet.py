
from __future__ import print_function, division

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, ZeroPadding2D, Add, Activation, Concatenate
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import optimizers
import numpy as np
from tqdm import tqdm

import tensorflow as tf

NonLinearActivation = 'relu' #try selu as well (besides relu)

def _expand(tensor111, dim1, dim2):
    """
    Expands a tensor to have dimensions dim1 and dim2
    """
    ans = K.repeat_elements(tensor111, dim1, 1)
    ans = K.repeat_elements(ans, dim2, 2)
    return ans

def BuildSUNet(convFunction, nFilter = 16, net_depth = 4, input_img_shape = (128,128,1), encoderSquash = 8, 
               innerConvs = 16, nBorderRefinementConvs = 3 ):
    """
    Builds shortcut U-Net model
    
    Assumes image and depth information already come in normalized
    
    nBorderRefinementConvs - number of convolutions after output has been computed (after last upsample)
    encoderSquash - Squash encoder outputs at each level to have this number of channels. Set to zero to skip squashing.
    """
    
    inputImg = Input(input_img_shape)
    inputDepth = Input( (1,1,1) )
    
    #s = Lambda(lambda x: x / 255) (inputs)

    #s = ReflectionPadding2D( padding = ((13, 14), (13, 14)) ) (s)
    #s = ZeroPadding2D( padding = ((13, 14), (13, 14)) ) (s)


    uLayers = []

    curLayer = inputImg
    curLayer = convFunction(curLayer, nFilter)
    for k in range(net_depth):
        curLayer = convFunction(curLayer, (2**(1+k))*nFilter)
        if encoderSquash > 0:
            squashedLayer = Conv2D(encoderSquash, (1,1), activation=None, padding='same') (curLayer)
            uLayers.append(squashedLayer)
        else:
            uLayers.append(curLayer)
            
        #curLayer = MaxPooling2D((2, 2)) (curLayer)
        curLayer = SeparableConv2D(nFilter, (3,3), strides=(2, 2), padding='same', activation=NonLinearActivation)(curLayer)
        

    #take depth into account in the deepest layer
    dim1 = input_img_shape[0] // (2**(net_depth))
    dim2 = input_img_shape[1] // (2**(net_depth))

    depthIn = Lambda(lambda x: _expand(x, dim1, dim2) )(inputDepth)
    curLayer = concatenate([curLayer, depthIn])
    
    
    curLayer = convFunction(curLayer, (2**net_depth)*nFilter)
    for k in range(innerConvs):
        ic = convFunction(curLayer, (2**net_depth)*nFilter)
        ic = convFunction(ic, (2**net_depth)*nFilter)
        curLayer = Add()([curLayer, ic])
    curLayer = ApplyDC4(curLayer, nFilter*net_depth, ksize = (3,3))    
    
    if encoderSquash > 0:
        curLayer = Conv2D(encoderSquash, (1,1), activation=None, padding='same') (curLayer)

    upsize = 2**net_depth
    curLayer = Conv2DTranspose(nFilter, (2, 2), strides=(upsize, upsize), padding='same') (curLayer)
    
    for k in range(net_depth):
        curULayer = uLayers[net_depth-k-1]
        upsize = 2**(net_depth-k-1)
        curULayer = Conv2DTranspose(nFilter, (2, 2), strides=(upsize, upsize), padding='same') (curULayer)
        curLayer = concatenate([curLayer, curULayer])

    #curLayer = ApplyDC4(curLayer, nFilter*net_depth, ksize = (3,3))    
    for k in range(nBorderRefinementConvs):
        curLayer = ApplyConv(curLayer, nFilter)
    

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (curLayer)
    
    #outputs = Cropping2D(cropping=((13, 14), (13, 14)) ) (outputs)

    model = Model(inputs=[inputImg, inputDepth], outputs=[outputs], name='SUNet_d{}'.format(net_depth) )
    
    #model.compile(optimizer='adam', loss=dice_loss, metrics=[mean_iou])
    
    return model

def BuildUNet(convFunction, nFilter = 8, net_depth = 4, input_img_shape = (128,128,1), encoderSquash = 0, nBorderRefinementConvs = 2 ):
    """
    Builds U-Net model
    
    Assumes image and depth information already come in normalized
    
    nBorderRefinementConvs - number of convolutions after output has been computed (after last upsample)
    encoderSquash - Squash encoder outputs at each level to have this number of channels. Set to zero to skip squashing.
    """
    
    inputImg = Input(input_img_shape)
    inputDepth = Input( (1,1,1) )
    
    #s = Lambda(lambda x: x / 255) (inputs)

    #s = ReflectionPadding2D( padding = ((13, 14), (13, 14)) ) (s)
    #s = ZeroPadding2D( padding = ((13, 14), (13, 14)) ) (s)


    uLayers = []

    curLayer = inputImg
    curLayer = convFunction(curLayer, nFilter)
    for k in range(net_depth):
        curLayer = convFunction(curLayer, (2**(1+k))*nFilter)
        if encoderSquash > 0:
            squashedLayer = Conv2D(encoderSquash, (1,1), activation=None, padding='same') (curLayer)
            uLayers.append(squashedLayer)
        else:
            uLayers.append(curLayer)
            
        curLayer = MaxPooling2D((2, 2)) (curLayer)

    #take depth into account in the deepest layer
    dim1 = input_img_shape[0] // (2**(net_depth))
    dim2 = input_img_shape[1] // (2**(net_depth))

    depthIn = Lambda(lambda x: _expand(x, dim1, dim2) )(inputDepth)
    curLayer = concatenate([curLayer, depthIn])
    
    curLayer = convFunction(curLayer, (2**net_depth)*nFilter)

    for k in range(net_depth):
        curLayer = Conv2DTranspose((2**(net_depth-k))*nFilter, (2, 2), strides=(2, 2), padding='same') (curLayer)
        curLayer = BatchNormalization() (curLayer)
        curLayer = concatenate([curLayer, uLayers[net_depth-k-1] ])
        curLayer = convFunction(curLayer, (2**(net_depth-k))*nFilter)
        #if k<depth-1:
        #   curLayer = ApplyAtt(curLayer, (2**(1+k))*nFilter)

    for k in range(nBorderRefinementConvs):
        curLayer = convFunction(curLayer, nFilter)
    

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (curLayer)
    
    #outputs = Cropping2D(cropping=((13, 14), (13, 14)) ) (outputs)

    model = Model(inputs=[inputImg, inputDepth], outputs=[outputs], name='UNet_d{}'.format(net_depth) )
    
    #model.compile(optimizer='adam', loss=dice_loss, metrics=[mean_iou])
    
    return model




# Define IoU metric, custom losses
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    smooth = 0.81
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_loss(y_true, y_pred):
    bin_crossentropyloss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    #bin_crossentropyloss = K.max(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return (1-dice_coef(y_true, y_pred))*0.99+bin_crossentropyloss*0.01

  
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 0.95, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def mean_iouKaggle(y_true, y_pred): #closer to Kaggle version
    scores = []
    batchSize = 32 #y_pred.get_shape().as_list()[0]
    #print(batchSize)
    for k in range(batchSize):
        y_pred_ = tf.to_int32(y_pred[k:k+1] > 0.5)
        score, up_opt = tf.metrics.mean_iou(y_true[k:k+1], y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        scores.append(score)

    return K.mean(K.stack(scores), axis=0)


class ReflectionPadding2D(ZeroPadding2D):
    def call(self, x, mask=None):
        pattern = [[0, 0],
                   [self.padding[0][0], self.padding[0][1]],
                   [self.padding[1][0], self.padding[1][1]],
                   [0, 0]]
        return tf.pad(x, pattern, mode='REFLECT')

def ApplyConv(s, outChannels, ksize = (3,3), padType = 'same' ):
    c1 = Conv2D(outChannels, ksize, activation=None, padding=padType) (s)
    c1 = BatchNormalization() (c1)
    c1 = Activation(NonLinearActivation)(c1)
    
    return c1

def ApplySepConv(s, outChannels, ksize = (3,3), padType = 'same' ):
    c1 = SeparableConv2D(outChannels, ksize, activation=None, padding=padType) (s)
    c1 = BatchNormalization() (c1)
    c1 = Activation(NonLinearActivation)(c1)
    
    return c1

def ApplyAtt(s, outChannels, ksize = (3,3), padType = 'same' ):
    c1 = non_local_block(s, mode='embedded', add_residual=True)
    return c1


def ApplyResNet(s, outChannels, ksize = (3,3), padType = 'same' ):
    s = Conv2D(outChannels, ksize, activation=None, padding=padType) (s)
    s = BatchNormalization() (s)
    s = Activation(NonLinearActivation)(s)
    
    c1 = Conv2D(outChannels, ksize, activation=None, padding='same') (s)
    c1 = BatchNormalization() (c1)
    c1 = Activation(NonLinearActivation)(c1)
    c1 = Add() ([s, c1])
    return c1

def ApplyInception(s, outChannels, ksize = (3,3), padType = 'same'  ):
    s = Conv2D(outChannels, ksize, activation=NonLinearActivation, padding=padType) (s)
    s = BatchNormalization() (s)
    nChannels = outChannels // 2
  
    #divide by 4 to try to match the total number of channels of other layer types
    tower_1 = Conv2D(nChannels, (5, 5), padding='same', activation=NonLinearActivation)(s)
    tower_1 = Conv2D(nChannels, (1,1), padding='same', activation=NonLinearActivation)(tower_1)
    tower_1 = BatchNormalization() (tower_1)
    
    tower_2 = Conv2D(nChannels, ksize, padding='same', activation=NonLinearActivation)(s)
    tower_2 = Conv2D(nChannels, (1, 1), padding='same', activation=NonLinearActivation)(tower_2)
    tower_2 = BatchNormalization() (tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(s)
    tower_3 = Conv2D(nChannels, (1, 1), padding='same', activation=NonLinearActivation)(tower_3)

    o = concatenate([tower_1, tower_2, tower_3], axis=3)
    o = Conv2D(outChannels, (1, 1), padding='same', activation=NonLinearActivation)(o)
    o = BatchNormalization() (o)
    return o

def ApplyDC4(s, outChannels, ksize = (3,3), padType = 'same'  ):
    s = Conv2D(outChannels, ksize, activation=NonLinearActivation, padding=padType) (s)
    s = BatchNormalization() (s)
    nChannels = outChannels // 4
  
    #divide by 4 to try to match the total number of channels of other layer types
    tower_1 = SeparableConv2D(nChannels, ksize, dilation_rate=(1, 1), padding='same', activation=NonLinearActivation)(s)
    
    tower_2 = SeparableConv2D(nChannels, ksize, dilation_rate=(2, 2), padding='same', activation=NonLinearActivation)(s)

    tower_3 = SeparableConv2D(nChannels, ksize, dilation_rate=(3, 3), padding='same', activation=NonLinearActivation)(s)

    tower_4 = SeparableConv2D(nChannels, ksize, dilation_rate=(4, 4), padding='same', activation=NonLinearActivation)(s)

    tower_5 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(s)

    o = concatenate([tower_1, tower_2, tower_3, tower_4, tower_5], axis=3)
    #o = Conv2D(outChannels, (1, 1), padding='same', activation=NonLinearActivation)(o)
    o = BatchNormalization() (o)
    return o

def ApplyDC(s, outChannels, ksize = (3,3), padType = 'same'  ):
    s = Conv2D(outChannels, ksize, activation=NonLinearActivation, padding=padType) (s)
    s = BatchNormalization() (s)
    nChannels = outChannels // 2
  
    #divide by 4 to try to match the total number of channels of other layer types
    tower_1 = Conv2D(nChannels, ksize, dilation_rate=(1, 1), padding='same', activation=NonLinearActivation)(s)
    #tower_1 = Conv2D(nChannels, (1, 1), padding='same', activation=NonLinearActivation)(tower_1)
    #tower_1 = BatchNormalization() (tower_1)
    
    tower_2 = Conv2D(nChannels, ksize, dilation_rate=(3, 3), padding='same', activation=NonLinearActivation)(s)
    #tower_2 = Conv2D(nChannels, (1, 1), padding='same', activation=NonLinearActivation)(tower_2)
    #tower_2 = BatchNormalization() (tower_2)

    tower_3 = Conv2D(nChannels, ksize, dilation_rate=(5, 5), padding='same', activation=NonLinearActivation)(s)
    #tower_3 = Conv2D(nChannels, (1, 1), padding='same', activation=NonLinearActivation)(tower_3)
    #tower_3 = BatchNormalization() (tower_3)

    o = concatenate([tower_1, tower_2, tower_3], axis=3)
    #o = Conv2D(outChannels, (1, 1), padding='same', activation=NonLinearActivation)(o)
    o = BatchNormalization() (o)
    return o

def ApplySDC(s, outChannels, ksize = (3,3), padType = 'same'  ):
    s = SeparableConv2D(outChannels, ksize, activation=NonLinearActivation, padding=padType) (s)
    s = BatchNormalization() (s)
    nChannels = outChannels // 2
  
    #divide by 4 to try to match the total number of channels of other layer types
    tower_1 = SeparableConv2D(nChannels, ksize, dilation_rate=(1, 1), padding='same', activation=NonLinearActivation)(s)
    tower_1 = BatchNormalization() (tower_1)
    
    tower_2 = SeparableConv2D(nChannels, ksize, dilation_rate=(2, 2), padding='same', activation=NonLinearActivation)(s)
    tower_2 = BatchNormalization() (tower_2)

    tower_3 = SeparableConv2D(nChannels, ksize, dilation_rate=(3, 3), padding='same', activation=NonLinearActivation)(s)
    tower_3 = BatchNormalization() (tower_3)

    o = concatenate([tower_1, tower_2, tower_3], axis=3)
    o = Conv2D(outChannels, (1, 1), padding='same', activation=NonLinearActivation)(o)
    o = BatchNormalization() (o)
    return o

def RDC(s, outChannels, ksize = (3,3)):
    s = ApplyDilatedConvs(s, outChannels, ksize)
    s = BatchNormalization() (s)
    s = Activation(NonLinearActivation)(s)
    
    c1 = ApplyDilatedConvs(s, outChannels, ksize)
    c1 = BatchNormalization() (c1)
    c1 = Activation(NonLinearActivation)(c1)
    c1 = Add() ([s, c1])
    return c1    


#Proper IOU evaluation Callback
import logging

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1, using_lovasz=False):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.score_list=[0]
        self.using_lovasz = using_lovasz

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            #first we have to threshhold
            t = 0.5
            if self.using_lovasz:
                t = 0
            IOU_list=[]
            for j in range(y_pred.shape[0]):
                y_pred_ = np.array(y_pred[j,:,:] > t, dtype=bool)
                y_val_=np.array(self.y_val[j,:,:], dtype=bool)
                
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

            prec_score=np.mean(prec_list)
            print('Accurate validation score is {}. Best so far is {}'.format(prec_score, max(self.score_list)))
            
            if prec_score > max(self.score_list):
                print('Saving model-tgs-salt-IV.h5')
                self.model.save('model-tgs-salt-IV.h5')
            
            self.score_list.append(prec_score)
# (snip)



# src: https://www.kaggle.com/aglotero/another-iou-metric
def IoU(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def IoUOld(a,b):
    intersection = ((a==1) & (a==b)).sum()
    union = ((a==1) | (b==1)).sum()
    if union > 0:
        return intersection / union
    elif union == 0 and intersection == 0:
        return 1
    else:
        return 0

    
from skimage.morphology import watershed
def WaterShedProc(img, lowThresh = 0.1, highThresh = 0.7):
    """
    Applies watershed thresholding to an img
    
    #receive img with shape [H,W,1]
    #returns img with shape [H,W,1]
    """
    markers = np.zeros_like(img[:,:,0])
    markers[img[:,:,0] < lowThresh] = 2
    markers[img[:,:,0] > highThresh] = 1
    segmentation = watershed(img[:,:,0], markers)
    segmentation[segmentation == 2] = 0
    segmentation = np.expand_dims(segmentation, axis=2)
    
    return segmentation

def WaterShedChangeAll(y_vals, lowThresh = 0.1, highThresh = 0.7):
    """
    Returns a watershed binarization of all y_vals
    """
    ans = np.zeros_like(y_vals)
    for k in tqdm(range(y_vals.shape[0])):
        ans[k] = WaterShedProc(y_vals[k], lowThresh, highThresh)
        
    return ans    
    
def non_local_block(ip, intermediate_dim=None, compression=4,
                    mode='embedded', add_residual=False):
    """
    Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).
    Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        compression: None or positive integer. Compresses the intermediate
            representation during the dot products to reduce memory consumption.
            Default is set to 2, which states halve the time/space/spatio-time
            dimension for the intermediate step. Set to 1 to prevent computation
            compression. None or 1 causes no reduction.
        mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or
            `concatenate`.
        add_residual: Boolean value to decide if the residual connection should be
            added or not. Default is True for ResNets, and False for Self Attention.
    Returns:
        a tensor of same shape as input
    """
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        intermediate_dim = channels // 2

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)

        size = K.int_shape(f)

        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    if add_residual:
        y = add([ip, y])

    return y


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x


############
# Fully Atrous
###########
def GetFullyAtrous():
    #try to do stuff without reducing the scale
    im_height = 101
    im_width = 101
    im_chan = 1


    inputImg = Input((im_height, im_width, im_chan))
    inputDepth = Input( (1,1,1) )

    #s = Lambda(lambda x: x / 255) (inputImg)
    s = inputImg

    dd = Lambda(lambda x: x * 0.001) (inputDepth)

    #s = ReflectionPadding2D( padding = ((13, 14), (13, 14)) ) (s)
    s = ZeroPadding2D( padding = ((13, 14), (13, 14)) ) (s)

    numFilters = 8
    s = ModelUNet.ApplyDC4(s, numFilters)
    s = ModelUNet.ApplyDC4(s, 2*numFilters)
    sc1 = s

    s = ModelUNet.ApplyDC4(s, 4*numFilters)
    sc2 = s

    s = ModelUNet.ApplyDC4(s, 8*numFilters)
    sc3 = s

    s = ModelUNet.ApplyDC4(s, 8*numFilters)

    s = ModelUNet.ApplyDC4(s, 8*numFilters)
    s = Add() ([s, sc3])

    s = ModelUNet.ApplyConv(s, 4*numFilters)
    s = Add() ([s, sc2])

    s = ModelUNet.ApplyConv(s, 2*numFilters)
    s = Add() ([s, sc1])

    s = ModelUNet.ApplyConv(s, 2*numFilters)
    s = ModelUNet.ApplyConv(s, 2*numFilters)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (s)


    outputs = Cropping2D(cropping=((13, 14), (13, 14)) ) (outputs)

    model = Model(inputs=[inputImg, inputDepth], outputs=[outputs])

    model.compile(optimizer='adam', loss=dice_loss, metrics=[mean_iou])

    model.summary()
    
    return model


##Lovasz Loss
def lovasz_loss(y_true, y_pred, per_image=True):
    y_true = K.cast(K.squeeze(y_true, -1), 'int32')
    y_pred = K.cast(K.squeeze(y_pred, -1), 'float32')
    logits = K.log(y_pred / (y_pred - 1.0))
    loss = lovasz_hinge(logits, y_true, per_image = per_image)
    return loss

def logits_lovasz_loss(y_true, y_pred, per_image=True):
    y_true = K.cast(K.squeeze(y_true, -1), 'int32')
    logits = K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    loss = lovasz_hinge(logits, y_true, per_image = per_image)
    return loss

"""
Lovasz-Softmax and Jaccard hinge loss in Tensorflow
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import tensorflow as tf
import numpy as np


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
        loss.set_shape((None,))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='all', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='all'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = 1
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = 1
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def keras_lovasz_softmax(labels,probas):
    #return lovasz_softmax(probas, labels)+binary_crossentropy(labels, probas)
    return lovasz_softmax(probas, labels)


