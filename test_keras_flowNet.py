from keras_flownet_model import k_model
from keras import backend as K
import numpy as np
import cv2
import os

def viz_flow(flow):
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


K.set_image_data_format('channels_last')
K.set_learning_phase(0)

model = k_model(shape=[512,512,6])
model.load_weights('keras_flownet.hdf5',by_name=True)

path = './imgs'
img_list = os.listdir(path)
img_list.sort(key=lambda x: int(x.split('.')[0]))
img_list = [os.path.join(path, i) for i in img_list]
for i in range(len(img_list) - 1):
    img_1 = cv2.imread(img_list[i])
    img_2 = cv2.imread(img_list[i + 1])
    images = np.array([cv2.resize(np.array(i), dsize=(512, 512)) for i in [img_1, img_2]])
    images_t = images.transpose(3, 0, 1, 2)  # [2,h,w,3]
    im = np.expand_dims(images_t.astype(np.float32),axis=0)
    rgb_mean = np.reshape(np.reshape(im,im.shape[:2]+(-1,)).mean(axis=-1),im.shape[:2] + (1,1,1))
    x = (im -rgb_mean) /255.0
    x = np.concatenate((x[:,:,0,:,:],x[:,:,1,:,:]),axis=1)
    x = np.transpose(x,[0,2,3,1])
    output = model.predict(x)[0]  # h,w,c


    bgr = viz_flow(output)
    cv2.imshow('ori', np.uint8(images[0] * 0.5 + images[1] * 0.5))
    cv2.imshow('bgr', bgr)
    cv2.waitKey(50)
