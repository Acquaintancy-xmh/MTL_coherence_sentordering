import torch
import utils
from utils import FLOAT, LONG, BOOL
import numpy

# avg_sents_repr = torch.zeros(10)
# avg_sents_repr = utils.cast_type(avg_sents_repr, FLOAT, use_gpu=False)
# order_score = torch.zeros(10)
# order_score = utils.cast_type(order_score, FLOAT, use_gpu=False)
# order_score[0] = int(avg_sents_repr[0].item())+3

# print(order_score)

a = []
a.append(torch.Tensor([1]).cuda())
a.append(torch.Tensor([2]).cuda())
a.append(torch.Tensor([3]).cuda())
a.append(torch.Tensor([4]).cuda())
# b = torch.Tensor([5,6,7,8])
print(a)
print(max(a))
print(sum(a) / len(a))

# a = [0.7738095238095238, 0.7202380952380952, 0.7388724035608308, 0.7032640949554896, 0.7359050445103857]

# print(numpy.mean(a))