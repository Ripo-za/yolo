from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "/home/ripo/project/python/workspace/cv/yolo/my/imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 3)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "/home/ripo/project/python/workspace/cv/yolo/my/cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "/home/ripo/project/python/workspace/cv/yolo/my/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()
    
args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes("/home/ripo/project/python/workspace/cv/yolo/my/data/coco.names")

#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
#断言式
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
"""
eval( )时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
"""
model.eval()

read_dir = time.time()
#Detection phase
try:
    # 没见过这么写的
    # osp.realpath('.') 打印出本地目录（”."指代本地目录的）的绝对路径
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()
    
if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]
# im_batches是处理过后的图像了吗
"""
这块因为loaded_ims 和 [inp_dim for x in range(len(imlist))]都是一列表那种，后面那个就是为了和前面的维度匹配
这块就是分别从他们俩取一个然后传入prep_image函数中。
"""
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
"""
这块为什么要反过来啊，把长宽的反着赋值
这块都宽和高调过来是因为在opencv里面图片的坐标是横着是x轴，竖着的是y轴，所以把长×宽---》宽×长就能对应了。
到后面话bounding box的时候就方便了
"""
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
"""
这块没必要循环
"""
# im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
im_dim_list = torch.FloatTensor(im_dim_list)

"""
im_dim_list 的每一行代表一个图片，有四列。两张图片的输出如下
tensor([[768., 576., 768., 576.],
        [773., 512., 773., 512.]])
"""

"""
 明白了
 这块是计算出入图片的batch_size大小。因为可能不会整除，所以最后一次有点特别
"""
leftover = 0
# 如果图片数不能整除 batch_size
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    # 这块是先算除法再算加法的，所以这个属于向上取整
    num_batches = len(imlist) // batch_size + leftover            
    # 这块是把batch_size个数的图片给竖着合并起来，这样一起传入进行检测。本程序就没有训练过程了。因为直接读取的权重文件。
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,len(im_batches))]))  for i in range(num_batches)]  

# The Detection Loop
write = 0

if CUDA:
    im_dim_list = im_dim_list.cuda()
    
start_det_loop = time.time()
"""
im_batches是一个四维的list。batch_num*图片信息
这个for循环用来前向传播得到最初始的bounding box，然后用NMS得到过滤后的bounding box，然后把所有的bounding box都放入output中，
用output的第一列数据区分这个bounding box 属于那个图片，这块加入了每个batch_size的偏移。
"""
for i, batch in enumerate(im_batches):
#load the image 
    """
    这里得到的batch是包含batch_size个数（最后一组除外）的图片信息。
    """
    start = time.time()
    if CUDA:

        batch = batch.cuda()
        # grad好像是代表梯度，但是我还不明白为啥这块这么写，应该是torch的原因
    with torch.no_grad():
        # 调用Darknet类的forward函数
        """
        这块返回的是三维的prediction 
        分别是[  batch_size  ,   每张图片所有预测的边框数  ，  每个bounding box的的属性（85列）  ]
        """
        prediction = model.forward(Variable(batch), CUDA)

   
    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)
    """
    这里返回的就是2维的了。进过NMS处理过后的prediction。维度信息分别是
    [本次batch_size个图片所有的bounding box数，每个bounding box的维度信息（batch_size内图片索引，***左上角坐标***，***右下角坐标***，置信度，属于这个类别的置信度，属于那个类别）]
    """

    end = time.time()
    # If the output of the write_results function for batch is an int(0),
    # meaning there is no detection, we use continue to skip the rest loop.
    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            # image.split("/")[-1]取照片的名字
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue
    """
    因为在write_result函数里，是以batch_size为主循环的，所以每个batch图片输出的时候，他返回的第一列图片的索引都是从0开始的
    这句的作用就是把列的图片索引加上每个batch_size的偏移大小
    通过这样做把检测的所有图片的bounding box信息都存储在了output里面。打破了batch_size之间的界限
    """
    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

    if not write:                      #If we have't initialised output
        output = prediction  
        write = 1
    else:
        # cat函数没有写第二个参数就是默认为0，这里是竖着合并
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        """
        这块判断if int(x[0]) == im_id 的原因是output中是每个边框一行，所以同一个图片可能有多个行（多个object），
        所以这里是把同一副图片的bound box全部挑出来的意思，然后用x[-1]这列判断是属于哪个类别。
        由于这是训练好的权重，所以是对应的。这块要注意。
        """
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]

        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()       
try:
    # 判断是否有识别的边框，如果所有的图像都没有object就退出，不用执行下一步的标注了
    """
    这里的output就是把predict的第一维给加上batch_size的偏移。就是能知道具体是哪张图片了。
    """
    output
    """ 
    output[所有的图片的bounding box数量，每个bounding box的信息]

    """
except NameError:
    print ("No detections were made")
    exit()


"""
--------------------------------------Drawing bounding boxes on images
"""
"""
------------torch.index_select()函数作用---------------
这个函数是根据第三个参数indices的内容来对第一个参数x来取数据，第二个参数是维度。这个函数可以对同一行重复的取。

x = torch.randn(3, 4)
print(x)
print(x.long())
indices = torch.tensor([0,0,2])
print(indices)
a=torch.index_select(x, 0, indices)
print(a)
b=torch.index_select(x, 1, indices)
print(b)
------------------------下面是输出----------------------
tensor([[ 1.3416,  0.0915,  1.0359,  0.9548],
        [ 0.0641,  2.0978,  0.7618,  1.6237],
        [ 0.2838, -0.4702,  0.5880,  0.0489]])
tensor([[1, 0, 1, 0],
        [0, 2, 0, 1],
        [0, 0, 0, 0]])
tensor([0, 0, 2])
tensor([[ 1.3416,  0.0915,  1.0359,  0.9548],
        [ 1.3416,  0.0915,  1.0359,  0.9548],
        [ 0.2838, -0.4702,  0.5880,  0.0489]])
tensor([[1.3416, 1.3416, 1.0359],
        [0.0641, 0.0641, 0.7618],
        [0.2838, 0.2838, 0.5880]])

--------------------tensor.long()函数就是强制转换为整行int64,向下取整=--------------
"""

"""  -----------------------------------------------------------------------------------------
接下里是对output表示的bounding box 的***左上角坐标***和***右下角坐标***进行处理，主要是因为最开始对图片进行预处理改变了大小。
所以这里要逆向输出一下，方便化bounding box。
"""
# 根据output的输出重复图片长宽信息
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
# 这块我把416改成了inp_dim，和网页上一样，这块为什么用min，是和前面处理图片有关系吗？？？？？？？？
"""
很多地方用对一维的tensor用view(-1,1)的目的是能把原来的[n]维变成[n,1]维！！！！！！！这点要注意
"""
print(torch.min(inp_dim/im_dim_list,1))
"""
我这块主要是没理解是min函数，这块的1是按行取最小值，而那个0最开始以为是取第一列的值，但其实是min函数返回两个值，上面说明过了
分别是最小值和最小值索引。这里我们要最小值，所以用[0]
"""
scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)

"""
这块不知道为啥要这样，没有道理啊。可能和图片预处理有关？？？？？？？
scaling_factor就是最开始图片预处理乘的因子，因为scaling_factor的计算方法与预处理时长宽乘的因子的计算方法一样。
长宽同时乘以一个因子来实现等比例缩放。
scaling_factor*im_dim_list[:,1]   和   scaling_factor*im_dim_list[:,0] 就是原始图片第一次预处理后的大小
然后用inp_dim减去他们两个的值在除以二，其中肯定有一个值为零（长和宽较大的那个）。然后另一个计算出来的就是把图片添加到canvas后
留出的空白地方的大小（预处理时用（128,128,128）填充的区域）

然后用output减去这些值得到偏移后的结果（有一个方向不用偏移）。
"""
output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
"""
最后再除以scaling_factor因子，复原成原来的。
"""
output[:,1:5] /= scaling_factor

"""
这块是把output[1:5]给转换成要画的矩形的顶点了。和计算IOU时很像
"""
# exit()
for i in range(output.shape[0]):
    # troch.clamp函数就是把所有的元素值，限定在指定范围内。第二三个参数分别是min和max。这里im_dim_list[i,0]应该是横向最大值
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("/home/ripo/project/python/workspace/cv/yolo/my/pallete", "rb"))

draw = time.time()


def write(x, results):
    """
    opencv的坐标是横着是x轴，竖着是y轴。
    """
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    # results是最开始用cv2读取出来的图像，因为它对应的实参是loaded_ims
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    # 这块开始写类别信息，首先获取将要写的文字要占多大地方
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    # 计算写类别的边框的大小
    c2 = (c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4)
    # 这里的-1是绘制填充矩形，这里的color是一个(R,G,B)
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 2), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


list(map(lambda x: write(x, loaded_ims), output))
# 这里应该是把imlist中每个元素都作用了lambda表达式的函数。
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))


end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
    
