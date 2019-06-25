from keras_flownet_model import k_model
from keras import backend as K
import numpy as np
import cv2
import os
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
from tensorflow.python.framework import graph_io

K.set_learning_phase(0)

model = k_model(shape=[None,None,6])
model.load_weights('keras_flownet.hdf5',by_name=True)

sess = K.get_session()
output_tensor = model.output
print(output_tensor)
outputs_node = [output_tensor.op.name]
print(outputs_node)
frozen_graph = convert_variables_to_constants(sess,sess.graph_def,outputs_node)

graph_io.write_graph(frozen_graph,'./','flowNet.pb',as_text=False)

