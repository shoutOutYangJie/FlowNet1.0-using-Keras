# FlowNet1.0-using-Keras
This model's weights are converted from Flownet of Nvidia

# Description
Because I restore model' parameters from Nvidia's FlowNet project, this repo doesn't support training. Note that I just convert simplest FlowNet: FlowNet-S.
But I thihk this repo is still helpful if you want to learn how to transfer the parameter from Pytorch to Keras. And I also provoid the srcipt which can generate Pb frozen graph.
Note that this repo includes test data, so You just run it, and you can look at the result.

# how to use
* download the original weights offered by [Nvidia](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)
* change the path of original weights in "convert,py"
> #13th line 

> dict = torch.load(r"F:\python\collision_detection\flownet2_pytorch\checkpoint/FlowNet2-SD_checkpoint.pth.tar")


* run "convert.py", and you will get "keras_flownet.hdf5" file.
> python convert.py

* run "convert_pb.py" to get "flowNet.pb" file.
> python convert_pb.py

* run "test_flowNet_pb.py" to check the result.
> python test_flowNet_pb.py

# Cite
[1] [Nvidia FlowNet](https://github.com/NVIDIA/flownet2-pytorch)
[2] [FlowNet2.0](https://arxiv.org/abs/1612.01925)
