
import cv2
import os
import tensorflow as tf
import numpy as np

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

pb_path = './flowNet.pb'
path = './imgs'
img_list = os.listdir(path)
img_list.sort(key=lambda x: int(x.split('.')[0]))
img_list = [os.path.join(path, i) for i in img_list]


with tf.gfile.FastGFile(pb_path,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name="")

graph = tf.get_default_graph()
input_name = 'input_1'
output_names = 'up_sampling2d_1/ResizeBilinear'

input_tensor = graph.get_tensor_by_name(input_name+':0')
score_output = graph.get_tensor_by_name(output_names+':0')

with tf.Session() as sess:
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

        output = sess.run(score_output,feed_dict={input_tensor:x})[0]
        print(output.shape)
        bgr = viz_flow(output)
        cv2.imshow('ori', np.uint8(images[0] * 0.5 + images[1] * 0.5))
        cv2.imshow('bgr', bgr)
        cv2.waitKey(50)