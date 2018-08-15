## import tensorflow as tf
#Normal Unet: 
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,Conv1D,SeparableConv2D,Lambda,ZeroPadding2D,Cropping2D,Concatenate,Bidirectional
from keras.layers import  CuDNNLSTM,Permute,Reshape,Flatten,Dense,Lambda,Multiply

from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation,Dropout
from keras import backend as K
from keras.layers.merge import concatenate
from keras.utils.data_utils import get_file
from keras.callbacks import Callback
from keras.callbacks import TensorBoard,LearningRateScheduler
from skimage.morphology import label
import numpy as np

def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=False))
# Attention UNET First the unet custom layer: It will take two inputs: The low level feature and the high level feature and it will returned a concatenation of the 
#an attention weightedhigh level and a low level 
#The Loop Projection Layer Transforms each Aggregation seperately

#we should write the attention layer as a function and only the parts where I need tf in lambda layers. toherwise we might have a problem with trainable layers



#basically we should define attention concat as a function and use custom Lmabda
#layer when needed
#Input: List High Level, high resoltion frst then low level low resolution
def Attention_Concat(inpu, feature_dim=None,feature_dim_2=None):
    cnn_1=Conv2D(feature_dim, (1, 1), padding='same')
    cnn_2=Conv2D(feature_dim, (1, 1), padding='same')

    cnn_3=Conv1D(feature_dim_2,  1,activation="relu", padding='same')
    cnn_4=Conv1D(1, 1,activation="sigmoid", padding='same')
    
    high_level=inpu[0]
    #means lower resolution and down in the unet. 
    low_level=inpu[1]
    size_big=int(high_level.shape[1])
    size_small=int(low_level.shape[1])

    #this should be the common dimension ( of the low leve)
    res=keras.layers.Reshape((size_small*size_small,feature_dim))
    con=keras.layers.Concatenate(axis=2)

    res_2=keras.layers.Reshape((size_small,size_small,1))
    up=UpSampling2DBilinear( (size_big,size_big))
    
    reduced_high_level=MaxPooling2D(pool_size=(2, 2))(high_level)

    hl_feat=cnn_1(reduced_high_level)
    ll_feat=cnn_2(low_level)
    
    #reshape them 
    hl_feat=res(hl_feat)
    ll_feat=res(ll_feat)

    con_feat=keras.layers.Concatenate(axis=2)([hl_feat,ll_feat])
    extra=cnn_3(con_feat)
    sigmoids=cnn_4(extra)
    #bring it back to 7x7x1
    sigmoids=res_2(sigmoids)
    sigmoids=up(sigmoids)
    
    weighted_hl=keras.layers.multiply([high_level,sigmoids])
    final=keras.layers.Concatenate( axis=3)([UpSampling2D(size=(2, 2))(low_level), weighted_hl])
    return final 
    

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 2
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1
# Pretrained weights
ZF_UNET_224_WEIGHT_PATH = 'https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/releases/download/v1.0/zf_unet_224.h5'


def preprocess_input(x):
    x /= 256
    x -= 0.5
    return x


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)



def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    #we also add the cat crossentropy as another loss. 
    return -dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    bin_crossentropyloss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    #bin_crossentropyloss = K.max(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return (1-dice_coef(y_true, y_pred))*0.3+bin_crossentropyloss*0.7


def double_conv_layer(x, size, dropout=0.1, batch_norm=True,kernel=3,dia=1):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (kernel, kernel),dilation_rate=dia, padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (kernel, kernel),dilation_rate=dia, padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv

import math
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.93
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop,  
            math.floor((1+epoch)/epochs_drop))
    
    if (lrate < 4e-5):
      lrate = 4e-5
    
    return lrate
lrate = LearningRateScheduler(step_decay)



def attention_unet_2(filters,k=3,d=1,fd=64):
    axis = 3
    dropout_val=0.1
    inputs = Input((128, 128, 1))
    #norm
    #s = Lambda(lambda x: x / 255) (inputs)
    #pad
    #s = ZeroPadding2D( padding = ((13, 14), (13, 14)) ) (s)
    
    
    #downsample path
    conv_224 = double_conv_layer(inputs, filters,kernel=5,dia=d)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters,kernel=5,dia=d)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters,kernel=5,dia=d)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters,kernel=3,dia=d)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters,kernel=3,dia=d)
    
    up_28 = Attention_Concat([conv_28,conv_14],feature_dim=fd*4,feature_dim_2=fd*2)
    up_conv_28 = double_conv_layer(up_28, 8*filters,kernel=3,dia=d)

    up_56 = Attention_Concat([conv_56,up_conv_28 ],feature_dim=fd*4,feature_dim_2=fd*2)
    up_conv_56 = double_conv_layer(up_56, 4*filters,kernel=3,dia=d)

    up_112 = Attention_Concat([conv_112,up_conv_56],feature_dim=fd*2,feature_dim_2=fd*2)
    up_conv_112 = double_conv_layer(up_112, 2*filters,kernel=5,dia=d)

    up_224 = Attention_Concat([conv_224,up_conv_112],feature_dim=fd,feature_dim_2=fd*2)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val,kernel=7,dia=d)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)

    conv_final = Activation('sigmoid')(conv_final)
    #conv_final = Cropping2D(cropping=((13, 14), (13, 14)) ) (conv_final)
    
    
    #create the model
    model = Model(inputs, conv_final, name="LL_UNET_ATN")
    
    model.compile(optimizer="adam", loss=dice_loss)

    return model 

def attention_unet_lstm(filters,k=3,d=1,fd=64):
    axis = 3
    dropout_val=0.1
    inputs = Input((128, 128, 1))
    inputs_lstm=Reshape((128,128))(inputs)
    
    #We aslo include a "high resolution" LSTM Path wiht attentionon which orientation 
    lstm_1=Bidirectional(CuDNNLSTM(units=128,return_sequences=True),merge_mode='sum')(inputs_lstm)
    lstm_2=Bidirectional(CuDNNLSTM(units=128,return_sequences=True)(Permute((2,1)),merge_mode='sum')(inputs_lstm))

    #downsample path
    conv_224 = double_conv_layer(inputs, filters,kernel=5,dia=d)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters,kernel=5,dia=d)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters,kernel=5,dia=d)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters,kernel=3,dia=d)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters,kernel=3,dia=d)
    
    #scaling the LSTM with Low level info
    for_softmax=Flatten()(conv_14)
    softmax=Dense(2,activation="softmax")(for_softmax)
    scales_lstm_1=Multiply()([Lambda(lambda x: x[:,0])(softmax),lstm_1])
    scales_lstm_2=Multiply()([Lambda(lambda x: x[:,1])(softmax),lstm_2])
    
    scales_lstm_1=Reshape((128,128,1))(scales_lstm_1)
    scales_lstm_2=Reshape((128,128,1))(scales_lstm_2)
    
    up_28 = Attention_Concat([conv_28,conv_14],feature_dim=fd*4,feature_dim_2=fd*2)
    up_conv_28 = double_conv_layer(up_28, 8*filters,kernel=3,dia=d)

    up_56 = Attention_Concat([conv_56,up_conv_28 ],feature_dim=fd*4,feature_dim_2=fd*2)
    up_conv_56 = double_conv_layer(up_56, 4*filters,kernel=3,dia=d)

    up_112 = Attention_Concat([conv_112,up_conv_56],feature_dim=fd*2,feature_dim_2=fd*2)
    up_conv_112 = double_conv_layer(up_112, 2*filters,kernel=5,dia=d)

    up_224 = Attention_Concat([conv_224,up_conv_112],feature_dim=fd,feature_dim_2=fd*2)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val,kernel=7,dia=d)

    final=Concatenate(axis=3)([scales_lstm_1,scales_lstm_2,up_conv_224])
    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(final)

    conv_final = Activation('sigmoid')(conv_final)
    #conv_final = Cropping2D(cropping=((13, 14), (13, 14)) ) (conv_final)
    
    
    #create the model
    model = Model(inputs, conv_final, name="LL_UNET_ATN")
    
    model.compile(optimizer="adam", loss=dice_loss)

    return model 


def conv_block(m, dim, acti, bn, res, do=0):
	n = Conv2D(dim, 3, activation=acti, padding='same')(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, padding='same')(n)
	n = BatchNormalization()(n) if bn else n
	return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
    dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    
    model=Model(inputs=i, outputs=o)
    model.compile(optimizer="adam", loss=dice_loss)

    return model 


def attention_unet(filters,k=3,d=1,fd=32):
    axis = 3
    dropout_val=0.1
    inputs = Input((101, 101, 2))
    #norm
    s = Lambda(lambda x: x / 255) (inputs)
    #pad
    s = ZeroPadding2D( padding = ((13, 14), (13, 14)) ) (s)
    
    
    #downsample path
    conv_224 = double_conv_layer(s, filters,kernel=k,dia=d)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters,kernel=k,dia=d)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters,kernel=k,dia=d)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters,kernel=k,dia=d)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters,kernel=k,dia=d)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters)
    
    #upsample path
    up_14 = Attention_Concat(feature_dim=fd*16,feature_dim_2=fd*16)([conv_14,conv_7])
    up_conv_14 = double_conv_layer(up_14, 16*filters,kernel=k,dia=d)

    up_28 = Attention_Concat(feature_dim=fd*8,feature_dim_2=fd*8)([conv_28,up_conv_14])
    up_conv_28 = double_conv_layer(up_28, 8*filters,kernel=k,dia=d)

    up_56 = Attention_Concat(feature_dim=fd*4,feature_dim_2=fd*4)([conv_56,up_conv_28 ])
    up_conv_56 = double_conv_layer(up_56, 4*filters,kernel=k,dia=d)

    up_112 = Attention_Concat(feature_dim=fd*2,feature_dim_2=fd*2)([conv_112,up_conv_56])
    up_conv_112 = double_conv_layer(up_112, 2*filters,kernel=k,dia=d)

    up_224 = Attention_Concat(feature_dim=fd,feature_dim_2=fd)([conv_224,up_conv_112])
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val,kernel=k,dia=d)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)

    conv_final = Activation('sigmoid')(conv_final)
    conv_final = Cropping2D(cropping=((13, 14), (13, 14)) ) (conv_final)
    
    
    #create the model
    model = Model(inputs, conv_final, name="LL_UNET_ATN")
    
    model.compile(optimizer="adam", loss=dice_loss)

    return model 



import tensorflow as tf
def le_norm(norm_list):
    original_shape=norm_list[0].shape
    #flatten it i didn trust normal flatten so id did it this way
    flat_list=[keras.layers.Reshape((int(original_shape[1]*original_shape[2]*original_shape[3]),))(y) for y in norm_list ]
    #stack together to normalize it 
    stack=keras.layers.Lambda(lambda x:K.stack(x,axis=2))(flat_list)
    #normalize it 
    norm=keras.layers.BatchNormalization(axis=1)(stack)
    #split it back 
    split=keras.layers.Lambda(lambda x:tf.split(x,num_or_size_splits=stack.shape[2],axis=2))(norm)
    #give back in original shapes 
    original=[keras.layers.Reshape((int(original_shape[1]),int(original_shape[2]),int(original_shape[3])))(y) for y in split]
    return original



def attention_unet_ensemb(filters,ensemb):
    axis = 3
    dropout_val=0.1
    inputs = Input((101, 101, 2))
    #norm
    s = Lambda(lambda x: x / 255) (inputs)
    #pad
    s = ZeroPadding2D( padding = ((13, 14), (13, 14)) ) (s)
    #downsample path
    norm_list_1=[]
    for j in range(0,ensemb):
        conv_224 = double_conv_layer(s, filters)
        pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)
        conv_112 = double_conv_layer(pool_112, 2*filters)
        norm_list_1.append(conv_112)
    norm_list_1=le_norm(norm_list_1)
    
    norm_list_2=[]
    for j in range(0,ensemb):
        pool_56 = MaxPooling2D(pool_size=(2, 2))(norm_list_1[j])
        conv_56 = double_conv_layer(pool_56, 4*filters)
        norm_list_2.append(conv_56)
    norm_list_2=le_norm(norm_list_2)
    
    
    norm_list_3=[]
    for j in range(0,ensemb):
        pool_28 = MaxPooling2D(pool_size=(2, 2))(norm_list_2[j])

        conv_28 = double_conv_layer(pool_28, 8*filters)
        norm_list_3.append(conv_28)
    norm_list_3=le_norm(norm_list_3)
    
    
    norm_list_4=[]
    for j in range(0,ensemb):
        pool_14 = MaxPooling2D(pool_size=(2, 2))(norm_list_3[j])

        conv_14 = double_conv_layer(pool_14, 16*filters)
        norm_list_4.append(conv_14)
    norm_list_4=le_norm(norm_list_4)
    
    norm_list_5=[]
    for j in range(0,ensemb):
        pool_7 = MaxPooling2D(pool_size=(2, 2))(norm_list_4[j])

        conv_7 = double_conv_layer(pool_7, 32*filters)
        norm_list_5.append(conv_7)
    norm_list_5=le_norm(norm_list_5)
    
    
    
    norm_list_6=[]
    for j in range(0,ensemb):
        up_14 = Attention_Concat(feature_dim=128,feature_dim_2=128)([norm_list_4[j],norm_list_5[j]])
        up_conv_14 = double_conv_layer(up_14, 16*filters)
        
        norm_list_6.append(up_conv_14)
    norm_list_6=le_norm(norm_list_6)
    
    norm_list_7=[]
    for j in range(0,ensemb):
        up_28 = Attention_Concat(feature_dim=128,feature_dim_2=128)([norm_list_3[j],norm_list_6[j]])
        up_conv_28 = double_conv_layer(up_28, 8*filters)
        
        norm_list_7.append(up_conv_28)
    norm_list_7=le_norm(norm_list_7)
    
    
    norm_list_8=[]
    for j in range(0,ensemb):
        up_56 = Attention_Concat(feature_dim=128,feature_dim_2=128)([norm_list_2[j],norm_list_7[j] ])
        up_conv_56 = double_conv_layer(up_56, 4*filters)
        
        norm_list_8.append(up_conv_56)
    norm_list_8=le_norm(norm_list_8)
   
    norm_list_9=[]
    for j in range(0,ensemb):
        up_112 = Attention_Concat(feature_dim=128,feature_dim_2=128)([norm_list_1[j],norm_list_8[j]])
        up_conv_112 = double_conv_layer(up_112, 2*filters)
        up_224 = Attention_Concat(feature_dim=128,feature_dim_2=128)([conv_224,up_conv_112])
        up_conv_224 = double_conv_layer(up_224, filters, dropout_val)
        norm_list_9.append(up_conv_224)
    norm_list_9=le_norm(norm_list_9)
   

    final_stack =keras.layers.Concatenate()(norm_list_9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (final_stack)
 

    conv_final = Cropping2D(cropping=((13, 14), (13, 14)) ) (outputs)
    
    
    #create the model
    model = Model(inputs, conv_final, name="LL_UNET_ATN")
    
    model.compile(optimizer="adam", loss=dice_loss)

    return model 




def my_iou(y_val,y_pred):
    t=0.5
    IOU_list=[]
    for j in range(y_pred.shape[0]):
        y_pred_ = np.array(y_pred[j,:,:] > t, dtype=bool)
        y_val_=np.array(y_val[j,:,:], dtype=bool)
        overlap = y_pred_*y_val_ # Logical AND
        union = y_pred_ + y_val_ # Logical OR
        IOU = overlap.sum()/float(union.sum())
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

    return np.mean(prec_list,dtype="float64")

def iou_metric(y_true_in, y_pred_in, print_table=False):
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


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)