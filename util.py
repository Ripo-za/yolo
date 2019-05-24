from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):
    # print(tensor.shape,tensor)
    tensor_np = tensor.detach().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    # print(prediction.size(0),prediction.size(1),prediction.size(2))
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    print(inp_dim,stride,"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    """
    这个grid_size不是格子的大小，
    其实可以说是格子的数量。也就是图片其实被分成grid_size×grid_size块，每块的边长是stride
    """
    grid_size = inp_dim // stride
    # grid_size = 52

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    #这个程序和我以前看的那个第一版yolo不一样，这个是B*(5+C),那个是B*5+C
    """
    这块这些值都是配置文件里计算好了的。就比如num_anchors，所以才能正好改变形状
    """
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size,grid_size*grid_size*num_anchors, bbox_attrs)
    # 以格子数为单位
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    #是否是一个对象的置信度
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # print(prediction[:,0:30:,:2])
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    # (-1,1)就是所有按列排列  n×1这样
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    #先按深度拼接（列，变成两列），然后repeat循环num_anchors次，变成2*num_anchors(本程序为3)=6列，然后在变形，然后在扩展一维变成三维的
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    """
    上面变完就是这样的了
    0 0
    0 0
    0 0
    1 0
    1 0
    1 0
    。
    。
    。
    。
    这种排列不是每个anchors在一起的排列方式，是交错的排列
    """
    # 计算占多少个格子，下面会乘每个格子的大小（步长）（stride其实就是步长）。
    """
    这块是我一直以来最疑惑的，因为我就不明白他怎么表示每一个单元格（我就纳闷怎么表示单元格，因为我以前的理解只能表示每个像素），
    其实这个就是个表示的问题，也就是我们认为的认为他是单元格，这个程序确实让我见识到了神经网络那种（无脑梯度下降的厉害）。
    这里的prediction我们上面把他的第二维设置的是grid_size*grid_size*num_anchors,这点尤为重要，这样我们就能把这个图片看作是
    有grid_size*grid_size个格子，并且每个格子能预测num_anchors个锚点框。
    这就解释了上面怎么来表示每个格子的问题。
    还有一点就是我们还认为prediction表示的格子是从(0,0)点开始的。其实上面解释了x_y_offset处理完之后是什么样的，其实并不是我们最初假设
    的那样，没事锚点框的预测是在一起的。你看他都是连续3个一样的坐标。所以这三个锚点框是交叉排布的。
    这块应该在编写损失函数的时候注意就好了。只要我们的损失函数表述正确就好。因为每个值其实都是没有意义的，是我们赋予了他意义。
    """
    prediction[:,:,:2] += x_y_offset
    
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    # print(anchors.shape)
    if CUDA:
        anchors = anchors.cuda()
    
    # 原来是3*2 然后行重复grid_size*grid_size倍,在用unsqueeze(0)扩充一维，得到和上面一样的维度都是grid_size×grid_size×num_anchors
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    

    #归一化类别的置信度
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
    # 最后都乘步长，也就是每个格子的大小，来计算坐标
    prediction[:,:,:4] *= stride

    return prediction


#The functions takes as as input the prediction, confidence (objectness score threshold), 
# num_classes (80, in our case) and nms_conf (the NMS IoU threshold).
def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # 这块把置信度小于阈值的设为0
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    # new函数的作用已经解释过了。主要是创建和prediction类型一样的tensor，具体维度用参数指定
    box_corner = prediction.new(prediction.shape)
    """
    0--->x
    1--->y
    2--->w
    3--->h
    (box_corner[:,:,0],box_corner[:,:,1])表示的是左上角点的坐标 
    (box_corner[:,:,2],box_corner[:,:,3])表示的是右下角点的坐标 
    """
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    """
    这块以前忽略了，从这里开始，所有的prediction[1:5]都是表示的是左上角和右下角的坐标了。不是中心点坐标和bounding box的长宽了。！！！
    """
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False
    


    for ind in range(batch_size):
        # 一张张的取图片
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
        """
        >>> a = torch.randn(4, 4)
        >>> a
        tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
                [ 1.1949, -1.1127, -2.2379, -0.6702],
                [ 1.5717, -0.9207,  0.1297, -1.8768],
                [-0.6172,  1.0036, -0.6060, -0.2432]])
        >>> torch.max(a, 1)
        torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
        """
        # 更具上面的例子可知，这是把每个的最大的索引以以为的形式返回了,1表示一行行的取最大值。
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        # 扩展维度是为了下面的cat函数进行合并
        max_conf = max_conf.float().unsqueeze(1)
        # 扩展维度是为了下面的cat函数进行合并
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        # 挑出是判断包含对象的 列的索引，因为第四列为零就是不包含对象了,这列的在predict_transform函数里用sigmoid处理过
        # 不对！！！！ 这个函数前两行就是把小于阈值的置信度设为0了
        non_zero_ind = torch.nonzero(image_pred[:,4])
        try:
            # squeeze是挤压，也就是降维，这里是降维1维了，unsqueeze就是反序列化，解释为升维。 然后把不为零的那行挑出来
            # 这里view(-1,7)有用吗
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        # 如果这个图片没有任何object信息（第五列都是0），就跳过。开始下一张图片
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image，获取每个框的类别
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index，上面说过了。
        # print(img_classes)

        # img_classes 是检测出来的所有类序号从小到大的排序，没有重复
        for cls in img_classes:
            #perform NMS

        
            # get the detections with one particular class
            # 这三句是把这类object的框都给提取出来
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            # 这个sort函数返回两个值第一个是值的排序，第二个是对应的索引，我们拿的是索引所以用[1]，这个只对
            # 这个函数是对
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            # 上面只是对image_pred_class的一列进行了排序，我们用这个索引把整个image_pred_class排序。
            # 相当于image_pred_class根据第五列排序，也就是置信度排序。
            
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections

            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                """
                因为这个idx的值是image_pred_class最初是的行数，但是我们在for循环里对image_pred_class进行了删除，
                所以很可能会越界，这里进行判断。越界就直接退出就好了。
                """
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break

                
                # unsqueeze(1)就是在第一维后面扩充  [24]--->[24,1]
                # 如果是unsqueeze(0)，就是[24]--->[1,24]
  
                """
                这块终于明白了，以前理解错了。
                这块是先按置信度排序，然后NMS的规则是如果两个bounding box 的IOU值大于阈值，则保留置信度高的那个bounding box，
                因为IOU值越大，说明bounding box的重合度越高。所以IOU要保留IOU值小的，因为可能这两个bounding box是别的是同一类的
                不同对象。
                因为我们在上面先对置信度做了排序，所以就直接筛选出IOU值小于阈值的bounding box所在行就行了。
                """
                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                # 这么一乘iou_mask中为零的会把image_pred_class对应的每一行变为零
                image_pred_class[i+1:] *= iou_mask       
                #Remove the non-zero entries
                # 上面语句执行后，所有
                # 这里用第四列的原因是，上面的已经把所有置信度为0的都消除了。所以这次再为0就可以确定是上面乘iou_mask导致的。
            
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                # print(image_pred_class)
            # new的作用是创建一个与image_pred_class类型相同的tensor，然后里面定义形状，这里定义为与image_pred_class行数相同
            # 以便链接，最后用ind作为值填充
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        # print(output)
        return output
        """
        这里返回的就是2维的了。进过NMS处理过后的prediction。维度信息分别是
        [本次batch_size个图片所有的bounding box数，每个bounding box的维度信息（batch_size内图片索引，***左上角坐标***，***右下角坐标***，置信度，属于这个类别的置信度，属于那个类别）]
        """

        
    except:
        return 0
    
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    # 按照最长的边来进行等比例缩放
    """
    比如img_w=600，img_h=1000   inp_dim=400
    那么new_w = 600*0.4=240  new_h=1000*0.4=400
    """
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    # 创建一个inp_dim×inp_dim×3维度的画布，然后全用128填充
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    # 这块是把上面等比例缩放的图片填入canvas中，如果原来的图片不是正方形的就会导致长宽不一致，边缘处已经用（128,128,128）填充了。
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    return canvas

def prep_image(img, inp_dim):
    # plt.imshow(img)
    """
    Prepare image for inputting to the neural network. 
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    """
    这块的转置是（2,0,1）是把原来的按照2,0,1排序，把（416,416,3）的转换为（3,416,416）
    但是我不同为什么要这么转置？？？？？？？？
    """
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    # div是归一化了吗？？？？？？？？？？？？？？？？？？？？？？？？s
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(namesfile):
    fp = open(namesfile, "r")
    # name = fp.readlines()
    # print(name)
    names = fp.read().split("\n")[:-1]
    return names
