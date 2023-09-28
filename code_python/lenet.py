#############################################导入包 ##################################################################################
####################################################################################################################################
import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
# 记录
from torch.utils.tensorboard import SummaryWriter
import logging


LOG_FORMAT = "%(asctime)s++++++%(message)s"

# 打印设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("////////////////",device)

# 定义输出日志与画图文件的名字
ScaleName="./plot/test"
LogName="./log/test.log"
logging.basicConfig(filename=LogName,level=logging.DEBUG, format=LOG_FORMAT)

####################################mnist数据集 做成小批量 放在一个迭代器中##########################################################################################
def load_data_mnist(batch_size):  #@save
    trans = [transforms.ToTensor()]
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(
        root="./data", train=False, transform=trans, download=True)
    print('训练集长度：',len(mnist_train))
    print('图片形状：',mnist_train[0][0].shape)
    print('lable：',mnist_train[0][1])
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
    data.DataLoader(mnist_test, batch_size, shuffle=False))
# 一批图片 256 
batch_size = 256
train_loader,test_loader = load_data_mnist(batch_size=batch_size)
# next(iter(mnist_train_iter)) ##查看数据
# for feature, lable in train_loader:
#     print(feature.shape,lable.shape)

############################################设置网络并且 查看每一层的情况##########################################################################################
################################################################################################################################################################################
class letnet(nn.Module):
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

# net = letnet().dnn
# ######## 输入一批256张图 输出256个预测结果  每个预测结果有10个概率
# X = torch.rand(size=(256, 1, 28, 28), dtype=torch.float32)

# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)



# ########################################每一轮的训练和测试函数
def train(epoch, model):
    train_loss = 0
    train_acc_num = 0
    # 模型变成训练模式
    model.train()
    for feature, lable in train_loader:
        # 标签以及图片放进Gpu
        feature = feature.to(device)
        lable = lable.to(device)
        # 上一轮的梯度置零
        optimizer.zero_grad()
        # 放入到网络中输出预测
        preds = model(feature)
        # 计算预测得到的损失
        loss = criterion(preds, lable)
        # 反向传播
        loss.backward()
        # 优化函数根据损失优化模型 更新参数
        optimizer.step()
        #  (preds.argmax(1) == lable).sum()   256个一批中预测结果概率最大的标签与lable一样的个数
        train_acc_num += (preds.argmax(1) == lable).sum()
        # train_acc_num 表示一次训练中 正确的所有的个数
        train_loss += loss.item() / len(train_loader)
    # 一次训练中 正确的所有的个数除于数据量 就是准确率
    train_acc = train_acc_num / len(train_loader.dataset)
    # 输出信息
    print(f'Epoch:{epoch:3} | Train Loss:{train_loss:6.4f} | Train Acc:{train_acc:6.4f}')

def test(model):
    test_acc_num = 0
    model.eval()
    with torch.no_grad():
        for feature, lable in test_loader:
            feature = feature.to(device)
            lable = lable.to(device)
            preds = model(feature)
            test_acc_num += (preds.argmax(1) == lable).sum()
    test_acc = test_acc_num / len(test_loader.dataset)
    print(f'Test Acc:{test_acc:6.4f}')
    return float(test_acc)

# ########################################3训练主函数  输入网络和多少轮
def Train(epochs, net, name):
    max = 0
    writer = SummaryWriter(ScaleName)
    logging.debug("/////////////////start training name:"+name+"/////////////////")  # 参数msg
    print("////////////////start training name:"+name+"//////////////////")  # 参数msg
    for epoch in range(epochs):
        # 训练
        train(epoch, net)
        # 测试输出
        x = test(net)
        writer.add_scalar(name,x,epoch)
        # 获取到最大的进行保存和打印
        if max < x:
            max = x
            print("max:", max, "epoch:", epoch)
            torch.save(net.state_dict(), "model_save/netmax_{}.pth".format(name))
            logging.debug("max:"+str(max)+"epoch:"+str(epoch))  # 参数msg
    writer.close()
    logging.debug("********************训练完毕****************")  # 参数msg

# # ############################
#####开始训练
net =letnet()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
Train(20, net,"3")


