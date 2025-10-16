import torch

# from surrogate_loss import SurrogateLossForRankingLoss
# from torch.nn import BCEWithLogitsLoss
from metrics import PrecisionAtK

true_y = [[1, -1 , -1, 1], [1, -1, 1, 1], [-1, -1, 1, 1]]
# true_y = [[1, -1 , -1, 1], [1, -1, 1, 1], [-1, -1, 1, 1]]
pred_y = [[1, -1 , -1, 1], [1, -1, 1, 1], [-1, -1, 1, 1]]



true_y = torch.tensor(true_y)
pred_y = torch.tensor(pred_y)

print(PrecisionAtK(pred_scores=pred_y, target_labels=true_y, k=3))

# true_y[true_y < 1] = 0

# Loss = SurrogateLossForRankingLoss(mode="u4")

# print(Loss.calcculate_cost_matrix(true_y))

# surrogate_loss = Loss.forward(pred_y, true_y)

# print(surrogate_loss)

# bce = BCEWithLogitsLoss()
# loss = bce(pred_y.float(), true_y.float())
# # loss = bce(pred_y, true_y)
# print(loss)
# print(bce(pred_y,true_y))



# import torch

# # 创建一个示例张量
# tensor = torch.tensor([[3.0, 1.0, 4.0, 1.5, 2.0], [1.0, 3.0, 4.0, 1.5, 2.0]])

# # 设置 k 的值
# k = 3

# # 使用 torch.topk 获取前 k 个最大值及其索引
# values, indices = torch.topk(tensor, k)

# print("前 k 个最大值：", values)
# print("对应的索引：", indices)

# # print(torch.index_select(tensor, 1, indices))

# # for i in range(tensor.size(0)):
# for i in range(tensor.shape[0]):
#     tmp_y = torch.index_select(tensor[i,:], 0, indices[i,:])
#     print(tmp_y)
#     print(torch.mean(tmp_y))


