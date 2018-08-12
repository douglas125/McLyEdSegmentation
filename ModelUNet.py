from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, ZeroPadding2D, Add, Activation
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import optimizers
import numpy as np

import tensorflow as tf

NonLinearActivation = 'relu' #try selu as well (besides relu)

def _expand(tensor111, dim1, dim2):
    """
    Expands a tensor to have dimensions dim1 and dim2
    """
    ans = K.repeat_elements(tensor111, dim1, 1)
    ans = K.repeat_elements(ans, dim2, 2)
    return ans

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
    smooth = 1.01
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_loss(y_true, y_pred):
    bin_crossentropyloss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) * 1e-2
    return (1-dice_coef(y_true, y_pred))+bin_crossentropyloss

  
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

def ApplyDC(s, outChannels, ksize = (3,3), padType = 'same'  ):
    s = Conv2D(outChannels, ksize, activation=NonLinearActivation, padding=padType) (s)
    s = BatchNormalization() (s)
    nChannels = outChannels // 2
  
    #divide by 4 to try to match the total number of channels of other layer types
    tower_1 = Conv2D(nChannels, ksize, dilation_rate=(1, 1), padding='same', activation=NonLinearActivation)(s)
    tower_1 = Conv2D(nChannels, (1, 1), padding='same', activation=NonLinearActivation)(tower_1)
    tower_1 = BatchNormalization() (tower_1)
    
    tower_2 = Conv2D(nChannels, ksize, dilation_rate=(2, 2), padding='same', activation=NonLinearActivation)(s)
    tower_2 = Conv2D(nChannels, (1, 1), padding='same', activation=NonLinearActivation)(tower_2)
    tower_2 = BatchNormalization() (tower_2)

    tower_3 = Conv2D(nChannels, ksize, dilation_rate=(3, 3), padding='same', activation=NonLinearActivation)(s)
    tower_3 = Conv2D(nChannels, (1, 1), padding='same', activation=NonLinearActivation)(tower_3)
    tower_3 = BatchNormalization() (tower_3)

    o = concatenate([tower_1, tower_2, tower_3], axis=3)
    o = Conv2D(outChannels, (1, 1), padding='same', activation=NonLinearActivation)(o)
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