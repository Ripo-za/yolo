from  __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 
from util import *




def parse_cfg(cfgfile):
    file = open(cfgfile,'r')
    # 这个还是和直接readlines有区别的
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] !='#']
    lines = [x.rstrip().lstrip()  for x in lines]
    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            # 把这一行两个 [  ]去掉要里面的
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            # 因为定义的时候可能会有空格，比如  a = 2,我们要添加  block[a]=2，要把等号两边的空格去掉
            block[key.rstrip()] = value.lstrip()
    
    blocks.append(block)
    return blocks


def create_modules(blocks):

    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    # 因为blocks第一层已经给了net_info
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if (x["type"] == "convolutional"):
            
            # 这块没懂  看后面就明白了，这块判断是否有batch_normalize，没有的话在convolutional层加bias
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            activation = x["activation"]
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size -1)//2
            else:
                pad = 0

            # add the convolutional layer
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(index),conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm{0}".format(index),bn)
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)   

        elif x["type"] == "upsample":
            stride = int(x["stride"]) 
            upsample = nn.Upsample(scale_factor=2,mode="bilinear")
            module.add_module("upsample_{}".format(index),upsample)
        
        elif x["type"] == "route":
            x["layers"] = x["layers"].split(",")
            # Start  of a route，可能有两个值
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
            
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)

def get_test_input():
    img = cv2.imread("/home/ripo/project/python/workspace/cv/yolo/my/dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    
    return img_


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):

    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info,self.module_list = create_modules(self.blocks)
        # print(self.net_info,self.module_list)
    
    def forward(self, x,CUDA):
        modules = self.blocks[1:]
        outputs = {}   # We cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            # print(x.shape)
            if module_type == "convolutional" or module_type == "upsample":
                # 这里每个模块都是nn.Sequential对象，可以直接传入x（输入）返回输出
                # print(x.shape)
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif module_type == 'yolo':        
                # 这块anchors是cfg文件yolo层选择的anchors  0 ,1 ,2 这样的
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                # Transform
                # pytorch 中tensor.data and tensor.detach()的区别在CSDN和CV收藏夹里都有
                # 在本程序中这句不重要，因为这句对训练起作用（反向传播），对前向传播没作用
                x = x.data  
                # anchors现在是个数组，包含三个（，）的元组
                """
                这里传入predict_transform函数的x就是上一层的输出
                """
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                # print(x.shape)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
                    
                else:       
                    # 这个因为是一个三维的每一行，第一位是batch，所以在第二维拼接就类相当于竖着拼接
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        
        return detections
    
    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        # print(weights)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    #确实有个别的convolutional（75）没有batch_normalize（73） 
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


# model = Darknet("/home/ripo/project/python/workspace/cv/yolo/my/cfg/yolov3.cfg")
# inp = get_test_input()
# model.load_weights("/home/ripo/project/python/workspace/cv/yolo/my/yolov3.weights")
# #这里能直接用model是因为这个类里面实现了__call__(),他会直接调用forward（）
# pred = model(inp, torch.cuda.is_available())

# write_results(pred,0.7,80)





"""
权重文件中权重的排列方式是

int32----------------5个--------
float32 ------------------------------
    batch_normalize 或 bias
    convolutional

前五个int32是一些信息。

后面的都是float32
然后落实到具体卷积中排列方式为
batch_normalize 或 bias
convolutional

其他的block，比如route、shutcut、yolo、upsample有的不需要权重，有的直接使用卷积层的输出



"""



"""
batch normal 
unsqueeze
"""
