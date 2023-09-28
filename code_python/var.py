#############################################导入包 ##################################################################################
####################################################################################################################################
import torch
from torch import nn
from torchvision import transforms

import cv2
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = torch.nn.Sequential(nn.Conv2d(1, 6, kernel_size=5,padding=2), nn.Sigmoid(),
                          nn.AvgPool2d(kernel_size=2, stride=2),
                          nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                          nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                          nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                          nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))
    def forward(self, x):
        out = self.dnn(x)
        return out


net = lenet()
# 加载训练号的模型
net.load_state_dict(torch.load('/media/zjh/gt500/hzl1080/lenet/model_save/netmax_3.pth'))
# 送进gpu
net=net.to(device)



################################################自己手写验证 处理图像########################################################################
####################################################################################################################################
####定义形状改变函数
# 3,234,255   rgb
###二值化 颜色反转
img = cv2.imread('./img/7.jpg',0)  #读取要预测的图片
# img = cv2.imread('1.jpg',0)
print("原图形状：",img.shape)  
# print("原图数据：",img)  
cv2.imwrite('img_redChannel.png',img)

# # 阈值函数   就是将图像的像素点经过阈值（threshold）比较重新设置为0或者255，  大于阈值的部分被置为255，小于部分被置为0
ret, img_binary = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
# print(img_binary)
# # 因为minist 训练集是黑底白字的  反转一下
img_binary=255-img_binary
# print("二值之后形状：",img_binary.shape)  #(388, 480, 3)  
# print("二值之后：",img_binary)  


# # 截取的时候不规范 变形
x_list =[]
y_list=[]
for i in range(0,img_binary.shape[0]):
    for j in range(0,img_binary.shape[1]):
        if img_binary[i,j] != 0:
            x_list.append(i)
            y_list.append(j)
    

# ######选出含有数字的区域
x_min =min(x_list)
x_max = max(x_list)
y_min =min(y_list)
y_max = max(y_list)

# 边缘多出一点
x_d = int((x_max-x_min)/5)
y_d = int((y_max-y_min)/5)

img_binary=img_binary[x_min-x_d:x_max+x_d, y_min-y_d:y_max+y_d]
print("二值剪切之后形状：",img_binary.shape)  #(172, 143, 3) 
# # 打印出来
cv2.imwrite('img_binary.png',img_binary)


img_binay_resize = cv2.resize(img_binary, (28, 28), interpolation=cv2.INTER_AREA)
print("二值剪切变小之后形状：",img_binay_resize.shape)  #(172, 143, 3) 
cv2.imwrite('img_resize.png',img_binay_resize)




transf = transforms.ToTensor()
my_mnist = transf(img_binay_resize)  ###转换成tensor类型
# 1,1,28,28
my_mnist =my_mnist.view(1, 1, 28, 28)
print(my_mnist.shape)
# # 最后要得到四维 (1, 1, 28, 28) 一张tensor类型的图片

# # 因为网络与图片要在同一个环境
input = my_mnist.to(device) ####送进去gpu


output = net(input)
# print(output)
# # gpu下的tensor不能直接转numpy，需要先转到cpu tensor后再转为numpy
# # 待转换类型的PyTorch Tensor变量带有梯度，直接将其转换为numpy数据将破坏计算图，因此numpy拒绝进行数据转换，实际上这是对开发者的一种提醒。如果自己在转换数据时不需要保留梯度信息，可以在变量转换之前添加detach()调用。
# sorted,indices=torch.sort(output,descending=True)
# print(sorted)
# print(indices[0][0])



# 另外一种方法
output = output.detach().cpu().numpy()
# # 因为结果输出是二维 取第一行也就是一批量中的第一个  现在输入的一批就是一张
pred = np.argmax(output[0])
print(pred)


# # 因为网络与图片要在同一个环境
# input = my_mnist.to(device) ####送进去gpu
# output = net(Variable(input))
# prob = F.softmax(output, dim=1)
# prob = Variable(prob)
# prob = prob.cpu().numpy()  #用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
# # print(prob[0])                #prob是10个分类的概率
# pred = np.argmax(prob[0]) #选出概率最大的一个
# print('预测结果',pred.item())
