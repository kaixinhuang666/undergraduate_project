from flask import Flask, render_template, request, json
from werkzeug.datastructures import FileStorage
import numpy as np
import cv2
import torch
from torch import nn
from torchvision import transforms

import cv2
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')


# class lenet(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.dnn=torch.nn.Sequential(nn.Conv2d(1, 6, kernel_size=5,padding=2),
#                           nn.Sigmoid(),
#                           nn.AvgPool2d(kernel_size=2, stride=2),
#                           nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
#                           nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
#                           nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
#                           nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))
#
#     def forward(self,x):
#         out=self.dnn(x)
#         return out
#
#
# def img_pretest(img_path):
#     img = cv2.imread(img_path, 0)
#     print("原图形状：", img.shape)
#     ret, img_bi = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
#     img_bi = 255 - img_bi
#     cv2.imwrite('./re.png', img_bi)
#     x_ls = []
#     y_ls = []
#     img_shape = img_bi.shape
#     for i in range(img_shape[0]):
#         for j in range(img_shape[1]):
#             if img_bi[i, j] != 0:
#                 x_ls.append(i)
#                 y_ls.append(j)
#
#     x_min = min(x_ls)
#     x_max = max(x_ls)
#     y_min = min(y_ls)
#     y_max = max(y_ls)
#     x_d = int((x_max - x_min) / 5)
#     y_d = int((y_max - y_min) / 5)
#     d = max(x_d, y_d)
#
#     xlen = x_max - x_min
#     ylen = y_max - y_min
#     max_len = max(xlen, ylen)
#     max_len = max_len + d
#     x_middle = int((x_max + x_min) / 2)
#     y_middle = int((y_max + y_min) / 2)
#     img_bi = img_bi[max(0, x_middle - int(max_len / 2)):min(x_middle + int(max_len / 2), img.shape[0]),
#              max(0, y_middle - int(max_len / 2)):min(y_middle + int(max_len / 2), img.shape[0])]
#     print("二值剪切之后形状：", img_bi.shape)
#     cv2.imwrite('img_binary.png', img_bi)
#     img_bi = cv2.resize(img_bi, (28, 28))
#     print("二值剪切变小之后形状：", img_bi.shape)
#     cv2.imwrite('./img_resize.png', img_bi)
#     trans = transforms.ToTensor()
#     img_beingrate = trans(img_bi)
#     img_beingrate = img_beingrate.view(1, 1, 28, 28)
#     return img_beingrate


# def predict_number(img_beingrate):
#     img_input = img_beingrate.to(device)
#     output = le(img_input)
#     import torch.nn.functional as F
#     output_softmax = F.softmax(output, dim=1).detach().cpu().numpy()[0]
#     pred = np.argmax(output_softmax)
#     print(pred)
#     prob = []
#     for i in range(10):
#         prob.append(round(output_softmax[i] / sum(output_softmax), 4))
#     return pred,prob


@app.route('/welcome')
def welcome():
    return render_template('welcome.html')


@app.route('/number_recognize',methods=['GET','POST'])
def predict():

    return render_template('number_recognize.html')


@app.route('/number_recognize_phone',methods=['GET','POST'])
def predict_phone():

    return render_template('number_recognize_phone.html')



# @app.route('/number_recognize_phone',methods=['GET','POST'])
# def predict_phone():
#     global pred,prob
#     if request.method=='POST':
#         # img_bytes=request.get_data()
#
#         # img_buffer_numpy = np.frombuffer(img_bytes, dtype=np.uint8)
#         # img_numpy = cv2.imdecode(img_buffer_numpy, 1)
#         # print(img_bytes,img_buffer_numpy,img_numpy)
#         img=request.files['image'].read()
#         print(type(img))
#         filename = 'img_test.png'
#
#         with open(filename, 'wb') as f:
#             f.write(img)
#             #return filename
#         path='./'+filename
#         imgbeingrate=img_pretest(path)
#         pred,prob=predict_number(imgbeingrate)
#         pred = int(pred)
#         print(pred,'\n',prob)
#         return render_template('number_recognize_phone.html', pred=pred, prob=prob)
#     return render_template('number_recognize_phone.html', pred=pred, prob=prob)


# @app.route('/try',methods=['GET','POST'])
# def try1():
#     # pred='None'
#     # prob=[]
#     global pred,prob
#     if request.method=='POST':
#         # img_bytes=request.get_data()
#
#         # img_buffer_numpy = np.frombuffer(img_bytes, dtype=np.uint8)
#         # img_numpy = cv2.imdecode(img_buffer_numpy, 1)
#         # print(img_bytes,img_buffer_numpy,img_numpy)
#         img=request.files['image'].read()
#         print(type(img))
#         filename = 'img_test.png'
#
#         with open(filename, 'wb') as f:
#             f.write(img)
#             #return filename
#         path='./'+filename
#         imgbeingrate=img_pretest(path)
#         pred,prob=predict_number(imgbeingrate)
#         print(pred,'\n',prob)
#         result=dict()
#         pred=int(pred)
#         result['result']=pred
#
#
#         for i in range(0,10):
#             result[str(i)]=float(prob[i])
#         print(result.keys(),result.keys())
#         result_json = json.dumps(result, ensure_ascii=False)
#         #prob_json=json.dumps(result, ensure_ascii=False)
#         # return render_template('try.html'),result_json
#         return render_template('try.html', pred=pred, prob=prob)
#     print(pred,prob)
#     return render_template('try.html', pred=pred, prob=prob)



if __name__ == '__main__':

    # le = lenet()
    # le.load_state_dict(torch.load('../model_save/netmax_try1.pth'))
    # le = le.to(device)
    # pred='None'
    # prob=[]



    app.run(host='0.0.0.0',debug=True)
