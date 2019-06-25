from models import FlowNet2SD
import torch
from torch import nn

from keras import layers as L
from keras import backend as K
from keras_flownet_model import k_model
K.set_image_data_format('channels_first')
K.set_learning_phase(0)

data = torch.randn(1,6,512,512)
t_model = FlowNet2SD()
dict = torch.load(r"F:\python\collision_detection\flownet2_pytorch\checkpoint/FlowNet2-SD_checkpoint.pth.tar")
t_model.load_state_dict(dict["state_dict"])
t_model.eval()
weights_from_torch = t_model.state_dict()
# f = open('pytorch_layers_name.txt', 'w')
for k,v in weights_from_torch.items():
    if 'bias' not in k:
        weights_from_torch[k] = v.data.numpy().transpose(2, 3, 1, 0)

# f.close()

# total_num = 0
# for n,p in t_model.named_parameters():
#     print(n)
#     print(p.size())   #  output_channels,input_channels, kernel_size,kernel_size
#                         # input_channels, output_channels, kernel_size, kernel_size
    # total_num += p.numel()
# print(total_num)  #45371666
# f = open('k_layers_names.txt','w')
k_model = k_model()
for layer in k_model.layers:
    current_layer_name = layer.name
    if current_layer_name=='conv0':
        weights = [weights_from_torch['conv0.0.weight'],weights_from_torch['conv0.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name=='conv1':
        weights = [weights_from_torch['conv1.0.weight'],weights_from_torch['conv1.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name=='conv1_1':
        weights = [weights_from_torch['conv1_1.0.weight'],weights_from_torch['conv1_1.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'conv2':
        weights = [weights_from_torch['conv2.0.weight'],weights_from_torch['conv2.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'conv2_1':
        weights = [weights_from_torch['conv2_1.0.weight'],weights_from_torch['conv2_1.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'conv3':
        weights = [weights_from_torch['conv3.0.weight'],weights_from_torch['conv3.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'conv3_1':
        weights = [weights_from_torch['conv3_1.0.weight'],weights_from_torch['conv3_1.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'conv4':
        weights = [weights_from_torch['conv4.0.weight'],weights_from_torch['conv4.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'conv4_1':
        weights = [weights_from_torch['conv4_1.0.weight'],weights_from_torch['conv4_1.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'conv5':
        weights = [weights_from_torch['conv5.0.weight'],weights_from_torch['conv5.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'conv5_1':
        weights = [weights_from_torch['conv5_1.0.weight'],weights_from_torch['conv5_1.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'conv6':
        weights = [weights_from_torch['conv6.0.weight'],weights_from_torch['conv6.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'conv6_1':
        weights = [weights_from_torch['conv6_1.0.weight'],weights_from_torch['conv6_1.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'deconv5':
        weights = [weights_from_torch['deconv5.0.weight'],weights_from_torch['deconv5.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'predict_flow6':
        weights = [weights_from_torch['predict_flow6.weight'],weights_from_torch['predict_flow6.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'upsampled_flow6_to_5':
        weights = [weights_from_torch['upsampled_flow6_to_5.weight'],weights_from_torch['upsampled_flow6_to_5.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'inter_conv5':
        weights = [weights_from_torch['inter_conv5.0.weight'],weights_from_torch['inter_conv5.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'deconv4':
        weights = [weights_from_torch['deconv4.0.weight'],weights_from_torch['deconv4.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'predict_flow5':
        weights = [weights_from_torch['predict_flow5.weight'],weights_from_torch['predict_flow5.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'upsampled_flow5_to4':
        weights = [weights_from_torch['upsampled_flow5_to_4.weight'],weights_from_torch['upsampled_flow5_to_4.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'inter_conv4':
        weights = [weights_from_torch['inter_conv4.0.weight'],weights_from_torch['inter_conv4.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'deconv3':
        weights = [weights_from_torch['deconv3.0.weight'],weights_from_torch['deconv3.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'predict_flow4':
        weights = [weights_from_torch['predict_flow4.weight'],weights_from_torch['predict_flow4.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'upsampled_flow4_to3':
        weights = [weights_from_torch['upsampled_flow4_to_3.weight'],weights_from_torch['upsampled_flow4_to_3.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'inter_conv3':
        weights = [weights_from_torch['inter_conv3.0.weight'],weights_from_torch['inter_conv3.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'deconv2':
        weights = [weights_from_torch['deconv2.0.weight'],weights_from_torch['deconv2.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'predict_flow3':
        weights = [weights_from_torch['predict_flow3.weight'],weights_from_torch['predict_flow3.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'upsampled_flow3_to2':
        weights = [weights_from_torch['upsampled_flow3_to_2.weight'],weights_from_torch['upsampled_flow3_to_2.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'inter_conv2':
        weights = [weights_from_torch['inter_conv2.0.weight'],weights_from_torch['inter_conv2.0.bias']]
        layer.set_weights(weights)
    elif current_layer_name == 'predict_flow2':
        weights = [weights_from_torch['predict_flow2.weight'],weights_from_torch['predict_flow2.bias']]
        layer.set_weights(weights)
    else :
        # if not (isinstance(layer,L.UpSampling2D) or isinstance(layer,L.Concatenate) or isinstance(layer,L.LeakyReLU) or isinstance(layer,L.Input)):
            print(layer)
            # raise ValueError('need to initialize this layer, but not founded in pytorch model')
k_model.save_weights('keras_flownet.hdf5')
