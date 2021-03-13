import torch
import utils
from utils import FLOAT, LONG, BOOL

avg_sents_repr = torch.zeros(10)
avg_sents_repr = utils.cast_type(avg_sents_repr, FLOAT, use_gpu=False)
order_score = torch.zeros(10)
order_score = utils.cast_type(order_score, FLOAT, use_gpu=False)
order_score[0] = int(avg_sents_repr[0].item())+3

print(order_score)

a = torch.Tensor([1,2,3,4])
b = torch.Tensor([5,6,7,8])
print(torch.cat((a, b), 0))