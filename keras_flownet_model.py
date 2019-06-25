import keras
from keras.models import Model
from keras import backend as K
from keras.layers import UpSampling2D, ZeroPadding2D, concatenate
from keras.layers import Conv2D, Input, LeakyReLU, Conv2DTranspose
import numpy as np
# K.set_image_data_format('channels_first')
K.set_learning_phase(0)

def k_model(shape):
    # x = Input(shape=[6,512,512],)
    x = Input(shape=shape, )
    conv0 = Conv2D(64,(3,3),padding='same',name='conv0')(x)
    conv0 = LeakyReLU(0.1)(conv0)
    padding = ZeroPadding2D()(conv0)
    conv1 = Conv2D(64,(3,3),strides=(2,2),padding='valid',name='conv1')(padding)
    conv1 = LeakyReLU(0.1)(conv1)
    conv1_1 = Conv2D(128,(3,3),padding='same',name='conv1_1')(conv1)
    conv1_1 = LeakyReLU(0.1)(conv1_1)
    padding = ZeroPadding2D()(conv1_1)
    conv2 = Conv2D(128,(3,3),strides=(2,2),padding='valid',name='conv2')(padding)
    conv2 = LeakyReLU(0.1)(conv2)
    conv2_1 = Conv2D(128,(3,3),padding='same',name='conv2_1')(conv2)
    conv2_1 = LeakyReLU(0.1)(conv2_1)
    padding = ZeroPadding2D()(conv2_1)
    conv3 = Conv2D(256,(3,3),strides=(2,2),padding='valid',name='conv3')(padding)
    conv3 = LeakyReLU(0.1)(conv3)
    conv3_1 = Conv2D(256,(3,3),padding='same',name='conv3_1')(conv3)
    conv3_1 = LeakyReLU(0.1)(conv3_1)
    padding = ZeroPadding2D()(conv3_1)
    conv4 = Conv2D(512,(3,3),strides=(2,2),padding='valid',name='conv4')(padding)
    conv4 = LeakyReLU(0.1)(conv4)
    conv4_1 = Conv2D(512,(3,3),padding='same',name='conv4_1')(conv4)
    conv4_1 = LeakyReLU(0.1)(conv4_1)
    padding = ZeroPadding2D()(conv4_1)
    conv5 = Conv2D(512,(3,3),strides=(2,2),padding='valid',name='conv5')(padding)
    conv5 = LeakyReLU(0.1)(conv5)
    conv5_1 = Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv5_1')(conv5)
    conv5_1 = LeakyReLU(0.1)(conv5_1)
    padding = ZeroPadding2D()(conv5_1)
    conv6 = Conv2D(1024,(3,3),strides=(2,2),padding='valid',name='conv6')(padding)
    conv6 = LeakyReLU(0.1)(conv6)
    conv6_1 = Conv2D(1024,(3,3),padding='same',name='conv6_1')(conv6)
    conv6_1 = LeakyReLU(0.1)(conv6_1)

    flow6 = Conv2D(2,(3,3),padding='same',name='predict_flow6')(conv6_1)
    flow6_up = Conv2DTranspose(2,(4,4),strides=(2,2),name='upsampled_flow6_to_5',padding='same')(flow6)
    deconv5 = Conv2DTranspose(512,(4,4),strides=(2,2),padding='same',name='deconv5')(conv6_1)
    deconv5 = LeakyReLU(0.1)(deconv5)

    # print(deconv5.get_shape())
    concat5 = concatenate([conv5_1,deconv5,flow6_up],axis=3)  # 16
    inter_conv5 = Conv2D(512,(3,3),padding='same',name='inter_conv5')(concat5)
    flow5 = Conv2D(2,(3,3),padding='same',name='predict_flow5')(inter_conv5)

    flow5_up = Conv2DTranspose(2,(4,4),strides=(2,2),name='upsampled_flow5_to4',padding='same')(flow5) #32
    deconv4 =  Conv2DTranspose(256,(4,4),strides=(2,2),name='deconv4',padding='same')(concat5)
    deconv4 = LeakyReLU(0.1)(deconv4)

    concat4 = concatenate([conv4_1,deconv4,flow5_up],axis=3)
    inter_conv4 = Conv2D(256,(3,3),padding='same',name='inter_conv4')(concat4)
    flow4 = Conv2D(2,(3,3),padding='same',name='predict_flow4')(inter_conv4)  # (1, 2, 32, 32)

    flow4_up = Conv2DTranspose(2,(4,4),strides=(2,2),name='upsampled_flow4_to3',padding='same')(flow4)  #64
    deconv3 = Conv2DTranspose(128,(4,4),strides=(2,2),name='deconv3',padding='same')(concat4)
    deconv3 = LeakyReLU(0.1)(deconv3)

    concat3 = concatenate([conv3_1,deconv3,flow4_up],axis=3)  # 64
    inter_conv3 = Conv2D(128,(3,3),padding='same',name='inter_conv3')(concat3)
    flow3 = Conv2D(2,(3,3),padding='same',name='predict_flow3')(inter_conv3)
    flow3_up = Conv2DTranspose(2,(4,4),strides=(2,2),name='upsampled_flow3_to2',padding='same')(flow3)  #128
    deconv2 = Conv2DTranspose(64,(4,4),strides=(2,2),name='deconv2',padding='same')(concat3)
    deconv2 = LeakyReLU(0.1)(deconv2)

    concat2 = concatenate([conv2_1,deconv2,flow3_up],axis=3)
    inter_conv2 = Conv2D(64,(3,3),padding='same',name='inter_conv2')(concat2)
    flow2 = Conv2D(2,(3,3),padding='same',name='predict_flow2')(inter_conv2)
    result = UpSampling2D(size=(4,4),interpolation='bilinear')(flow2)   # 4*128
    model = Model(x,result)
    return model



if __name__=='__main__':
    model = k_model([512,512,6])
    # keras.layers.Conv2D().set_weights()
    for l in model.layers:
        # print(type(l))
        print(l.name)
        # print(l)
        for i, w in enumerate(l.get_weights()):
            print('%d'%i  , w.shape)
        # print('**********************************')
        # 0(4, 4, 512, 1024)
        # 1(512, )

    # data = np.random.randn(1,6,512,512)
    # output = model.predict(data,verbose=1)

    # model.summary()   # 45,371,666
    # for w in model.get_weights():
    #     print(w.shape)  # kernel_size,kernel_size,input_channel,output_channel
                            # for deconv, [k,k, output_channel, input_channel]
    # model.get_layer()
    # model.save()
    # model.




