import torch
from torch.autograd import grad
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from Nets import CNNMnist



#loss对model的hessian matrix
def hessian(model,data,target):

    output=model(data)
    loss = F.nll_loss(output, target, weight=None, reduction='mean')

    #first_grad为loss对mdele.parameters()的一阶导，type为tuple
    first_grad=grad(loss,list(model.parameters()),create_graph=True)

    """要求hessian矩阵，需要first_grad的每一个元素对model.parameters()求导"""

    #方便起见，将模型的二阶导存到一个list
    second_grad=[]
    print(first_grad[0].size())
    for i in first_grad:

        #首先求出每个tensor中所含参数的个数
        temp=1
        for j in i.size():
            temp *= j

        #将参数矩阵转化为向量,x中的元素和i共享内存
        x=i.view([temp])

        #逐个对model.parameters()进行求导
        for i in range(len(x)):
            grad_=grad(x[i],list(model.parameters()),retain_graph=True)
            second_grad.append(grad_)

    return second_grad


if __name__=="__main__":


    #设置gpu
    gpu=-1
    device=torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')

    #加载数据
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset= datasets.MNIST('data', train=True, download=True, transform=trans_mnist)
    train_set=torch.utils.data.DataLoader(dataset, batch_size=64)

    net=CNNMnist().to(device)

    data, target= train_set.dataset[0]
    data = train_set.collate_fn([data]).to(device)
    target= train_set.collate_fn([target]).to(device)

    second_grad=hessian(net,data,target)
