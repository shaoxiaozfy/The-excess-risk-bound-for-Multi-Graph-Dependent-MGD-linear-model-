import torch
from sklearn.metrics import roc_auc_score


def HammingLoss(pred_labels, target_labels):
    '''
    Computing Hamming loss

    Parameters
    ----------
    pred_labels : Tensor
        MxQ Tensor storing the predicted labels of the classifier, if the ith 
        instance belongs to the jth class, then pred_labels[i,j] equals to +1, 
        otherwise pred_labels[i,j] equals to 0.
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    hammingloss : float
    ''' 
    return torch.mean((pred_labels != target_labels).float()).item()

def SubsetLoss(pred_labels, target_labels):
    '''
    Computing Subset loss

    Parameters
    ----------
    pred_labels : Tensor
        MxQ Tensor storing the predicted labels of the classifier, if the ith
        instance belongs to the jth class, then pred_labels[i,j] equals to +1,
        otherwise pred_labels[i,j] equals to 0.
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the
        jth class, then pred_labels[i,j] equals to +1, otherwise
        pred_labels[i,j] equals to 0.

    Returns
    -------
    Subsetloss : float
    '''

    # num_sample = pred_labels.size(0)
    #
    # correct = 0
    #
    # for i in range (num_sample):
    #     if torch.equal(pred_labels[i], target_labels[i]):
    #         correct = correct + 1

    return torch.mean(torch.max((pred_labels != target_labels).float(), dim=1)[0]).item()
    
def RankingLoss(pred_scores, target_labels):
    '''
    Computing ranking loss

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    rankingloss : float
    '''
    _, index = torch.sort(pred_scores, 1, descending=True)
    _, order = torch.sort(index, 1)
    has_label = target_labels == 1
    
    rankingloss = 0.0
    count = 0
    num_data, num_classes = pred_scores.size()
    for i in range(num_data):
        m = torch.sum(has_label[i,:]).item()
        n = num_classes - m
        if m != 0 and n != 0:
            rankingloss = rankingloss + (torch.sum(order[i, has_label[i, :]]).item()
                                         - m*(m-1)/2.0) / (m*n)
            count += 1
            
    return rankingloss / count

def MicroAUC(pred_scores, target_labels):
    true_y, predicted_y = eliminate_all_zero_columns(target_labels, pred_scores)
    true_y, predicted_y = eliminate_all_one_columns(true_y, predicted_y)
    if len(true_y) != len(predicted_y):
        raise ValueError("Size mismatch for true_y and predicted_y tensors")

    # f1 = multiclass_f1_score(predicted_y, true_y, num_classes=81+81+21)
    # auprc = multilabel_auprc(predicted_y, true_y, average="macro")
    for i in range(true_y.shape[1]):
        if len(torch.unique(true_y[:, i])) != 2:
            print(true_y[:, i])
            raise ValueError(
                "Only one class present in y_true. ROC AUC score "
                "is not defined in that case."
            )
    if true_y.shape[1] > 0:
        # print("true_y is Nan:{}".format(torch.isnan(true_y).any()))
        # print("predicted_y is Nan:{}".format(torch.isnan(predicted_y).any()))
        true_y_numpy = true_y.cpu().numpy()
        predicted_y_numpy = predicted_y.cpu().numpy()
        roc_auc = roc_auc_score(true_y_numpy, predicted_y_numpy, average="macro", multi_class="ovo")
        #roc_auc = roc_auc_score(true_y_numpy, predicted_y_numpy, average="micro", multi_class="ovr")
        return roc_auc

def PrecisionAtK(pred_scores, target_labels, k=1):
    values, indices = torch.topk(pred_scores, k)
    precision_at_k = 0
    num_instance = pred_scores.shape[0]
    target_labels[target_labels < 1] = 0

    for i in range(num_instance):
        tmp_y = torch.index_select(target_labels[i,:], 0, indices[i,:])
        precision_at_k = precision_at_k + torch.mean(tmp_y.float())
    
    precision_at_k = precision_at_k / num_instance
    return precision_at_k

def eliminate_all_one_columns(y_true,y_pred):
    retain_indexs = []
    for i in range(y_true.shape[1]):
        if len(torch.unique(y_true[:, i])) == 2:
            retain_indexs.append(i)
    if len(retain_indexs) == y_true.shape[1]:
        return y_true,y_pred
    else:
        a,b = y_true[:,retain_indexs],y_pred[:,retain_indexs]
    return a,b

def eliminate_all_zero_columns(y_true,y_pred):
    indicators = torch.nonzero(torch.sum(y_true, dim=0))
    # indictors = torch.any(y_true.bool(),dim=0) #

    if indicators.shape[0] > 0:
        y_true_list = torch.split(y_true, 1, 1)
        y_pred_list = torch.split(y_pred, 1, 1)

        y_true_tmp = [i for num, i in enumerate(y_true_list) if num in indicators]
        y_pred_tmp = [i for num, i in enumerate(y_pred_list) if num in indicators]
        if indicators.shape[0] > 1:
            y_true,y_pred = torch.concat(y_true_tmp,1),torch.concat(y_pred_tmp,1)
        elif indicators.shape[0] == 1:
            y_true,y_pred = y_true_tmp[0],y_pred_tmp[0]
    elif indicators.shape[0] == 0:
        raise NotImplementedError

    return y_true,y_pred


