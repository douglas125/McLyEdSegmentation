from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tqdm import tqdm
from scipy.signal import medfilt2d
import os
import numpy as np



def ReadSegmentationImages(folder, depthPd = None, readMasks = True, applyMedianFilter = False, 
                           im_height=101, im_width=101):
    """
    Reads images from a folder.
    
    Expects images with dimensions width, height
    depthPd - Pandas dataframe containing ids and corresponding depths, with entry 'id', and depth is 'z'
    desiredStd - desired std dev. if zero, recomputes and returns mean and stddev
    
    
    Expected structure:
    folder/images
    folder/masks if readMasks = True
    
    returns images, masks, depths (if depthPd not None)
    depths are images with just 1 pixel (to use in imagegenerator from keras with the same randomization when generating)
    """
    
    imgFolder = os.path.join(folder, 'images')
    maskFolder = os.path.join(folder, 'masks')
    imgIds = next(os.walk(imgFolder))[2]
    
    list_length = len([img for img in imgIds if img not in ignoreList])
    ignored_length = len([img for img in ignoreList if img in imgIds])

    src_images = np.zeros((list_length, im_height, im_width, 1), dtype=np.uint8)
    
    if readMasks:
        src_masks = np.zeros((list_length, im_height, im_width, 1), dtype=np.bool)
    
    if depthPd is not None:
        depths = np.zeros( (list_length, 1,1,1) , dtype=np.float32)
        
    print('Getting images and masks ... ')
    for n, id_ in tqdm(enumerate([imId for imId in imgIds if imId not in ignoreList]), total=list_length):
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

    print('Ignored {} files'.format(len(ignoreList)))
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

        
        
ignoreList = ['05b69f83bf.png',
 '0d8ed16206.png',
 '1019e6ed47.png',
 '10833853b3.png',
 '135ae076e9.png',
 '1a4ffa4415.png',
 '1a7f8bd454.png',
 '1b0d74b359.png',
 '1c0b2ceb2f.png',
 '1efe1909ed.png',
 '1f0b16aa13.png',
 '1f73caa937.png',
 '20ed65cbf8.png',
 '287b0f197f.png',
 '2fb6791298.png',
 '37df75f3a2.png',
 '3a7bbd73af.png',
 '3c1d07fc67.png',
 '3ee4de57f8.png',
 '3ff3881428.png',
 '40ccdfe09d.png',
 '423ae1a09c.png',
 '4f30a97219.png',
 '51870e8500.png',
 '573f9f58c4.png',
 '58789490d6.png',
 '590f7ae6e7.png',
 '5aa0015d15.png',
 '5edb37f5a8.png',
 '5fabfdc0b6.png',
 '5ff89814f5.png',
 '6b95bc6c5f.png',
 '6f79e6d54b.png',
 '755c1e849f.png',
 '762f01c185.png',
 '7769e240f0.png',
 '808cbefd71.png',
 '859f623d32.png',
 '8c1d0929a2.png',
 '8ee20f502e.png',
 '9260b4f758.png',
 '96049af037.png',
 '96d1d6138a.png',
 '974f5b6876.png',
 '97515a958d.png',
 '99909324ed.png',
 '9aa65d393a.png',
 'a2b7af2907.png',
 'a31e485287.png',
 'a3e0a0c779.png',
 'a48b9989ac.png',
 'a536f382ec.png',
 'a56e87840f.png',
 'a7a87180a3.png',
 'a8be31a3c1.png',
 'a9e940dccd.png',
 'a9fd8e2a06.png',
 'aa97ecda8e.png',
 'acb95dd7c9.png',
 'ad2113ea1c.png',
 'b11110b854.png',
 'b552fb0d9d.png',
 'b637a7621a.png',
 'b8c3ca0fab.png',
 'b9bf0422a6.png',
 'bcae7b57ec.png',
 'bedb558d15.png',
 'c1c6a1ebad.png',
 'c20069b110.png',
 'c3589905df.png',
 'c79d52ffc3.png',
 'c8404c2d4f.png',
 'cc15d94784.png',
 'd0244d6c38.png',
 'd0e720b57b.png',
 'd1665744c3.png',
 'd2e14828d5.png',
 'd6437d0c25.png',
 'd8bed49320.png',
 'd93d713c55.png',
 'dcca025cc6.png',
 'de95e861ac.png',
 'e0da89ce88.png',
 'e51599adb5.png',
 'e7da2d7800.png',
 'e82421363e.png',
 'ec542d0719.png',
 'f0190fc4b4.png',
 'f26e6cffd6.png',
 'f2c869e655.png',
 'f9fc7746fb.png',
 'ff9d2e9ba7.png']