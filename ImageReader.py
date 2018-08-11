from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm
from scipy.signal import medfilt2d
import os
import numpy as np



def ReadSegmentationImages(folder, depthPd = None, readMasks = True, applyMedianFilter = False, im_height=101, im_width=101):
    """
    Reads images from a folder.
    
    Expects images with dimensions width, height
    depthPd - Pandas dataframe containing ids and corresponding depths, with entry 'id', and depth is 'z'
    
    
    Expected structure:
    folder/images
    folder/masks if readMasks = True
    
    returns images, masks, depths (if depthPd not None)
    depths are images with just 1 pixel (to use in imagegenerator from keras with the same randomization when generating)
    """
    
    imgFolder = os.path.join(folder, 'images')
    maskFolder = os.path.join(folder, 'masks')
    imgIds = next(os.walk(imgFolder))[2]
    
    src_images = np.zeros((len(imgIds), im_height, im_width, 1), dtype=np.uint8)
    
    if readMasks:
        src_masks = np.zeros((len(imgIds), im_height, im_width, 1), dtype=np.bool)
    
    if depthPd is not None:
        depths = np.zeros( (len(imgIds), 1,1,1) , dtype=np.float32)
        
    print('Getting images and masks ... ')
    for n, id_ in tqdm(enumerate(imgIds), total=len(imgIds)):
        img = load_img(  os.path.join(imgFolder, id_) )
        x = img_to_array(img)

        if depthPd is not None:
            depthVal = depthPd[depthPd['id'] == id_.split('.')[0]].z.tolist()
            if len(depthVal) == 1:
                depths[n,0,0,0] = depthVal[0]
            else:
                depths[n,0,0,0] = 0
                print('Depth not found for {}'.format(id_))


        x = x[:,:,0]

        #try median filtering
        if applyMedianFilter:
            x = medfilt2d(x)
            
        x = x.reshape( (im_height,im_width,1) )

        src_images[n] = x[:,:,0:1]
        if readMasks:
            mask = img_to_array(load_img( os.path.join(maskFolder, id_) ))
            src_masks[n] = mask[:,:,0:1]

    if depthPd is not None:
        if readMasks:
            return src_images, src_masks, depths
        else:
            return src_images, depths
    else:
        if readMasks:
            return src_images, src_masks
        else:
            return src_images
