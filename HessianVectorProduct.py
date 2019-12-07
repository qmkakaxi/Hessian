import torch
from torch.autograd import grad
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from Nets import CNNMnist
import six
import copy


#loss对model的hessian vector product,海森矩阵*v
def hvp(model, data, target,v):

    output=model(data)
    loss = F.nll_loss(output, target, weight=None, reduction='mean')

     #first_grad为loss对mdele.parameters()的一阶导，type为tuple
    first_grads = grad(loss, list(model.parameters()),  create_graph=True)
    grad_v = 0

    """hessian vector product"""

    for g, v in six.moves.zip(first_grads, v):
        grad_v += torch.sum(g * v)
    hvp=grad(grad_v, list(model.parameters()), create_graph=True)

    return hvp





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
    v=copy.deepcopy(list(net.parameters()))
    hvp=hvp(net,data,target,)
