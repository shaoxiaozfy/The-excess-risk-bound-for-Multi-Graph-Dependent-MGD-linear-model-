import numpy as np
import matplotlib.pyplot as plt
from utils.get_path import get_project_path
import os
import seaborn as sns


def read_training_log(file_name):
    log_dir = get_project_path() + file_name
    with open(log_dir) as f:
        lines = f.readlines()
    metrics,losses = [],[]

    for line in lines:
        if "Top1_MacroAUC_Epoch" in line:
            num = float(line.split("=")[-1].strip())
            metrics.append(num)
        elif "Loss_Epoch" in line:
            l = float(line.split("=")[-1].strip())
            losses.append(l)

    for i in range(len(metrics)-1):
        assert len(metrics) == len(losses)
    return metrics,losses

def read_test_log(file_name):
    log_dir = get_project_path() + file_name
    with open(log_dir) as f:
        lines = f.readlines()

    metrics,losses, forgets = [],[], []
    for line in lines:
        if "Top1_MacroAUC_Stream" in line:
            num = float(line.split("=")[-1].strip())
            metrics.append(num)
        elif "Loss_Stream" in line:
            l = float(line.split("=")[-1].strip())
            losses.append(l)
        elif "StreamForgetting" in line:
            f = float(line.split("=")[-1].strip())
            forgets.append(f)

    assert len(metrics) == len(losses) == len(forgets)
    return metrics, losses, forgets

def train_plot(metrics,losses,metrics2, losses2):
    sns.set_style("dark")
    fig = plt.figure(figsize=(9,5))
    fig1 = plt.subplot(121)

    plt.plot(list(range(len(metrics))), metrics,color="red",label="u_2")
    plt.plot(list(range(len(metrics))), metrics2, color="tab:blue",label="u_1")

    font1 = {'size': 14,}

    # ax1.set_ylim((-0.05,1.19))
    plt.xlabel("epochs",font1)#fill the meaning of X axis
    plt.ylabel("Micro-AUC",font1,)#fill the meaning of Y axis
    plt.legend(loc="lower right")

    # ax2.set_ylim((-0.42, 4.9))
    fig2 = plt.subplot(122)
    plt.plot(list(range(len(losses))), losses,color="red",label="u_2")
    plt.plot(list(range(len(losses))), losses2, color="tab:blue",label="u_1")

    plt.xlabel("epochs", font1)
    plt.ylabel('Loss', font1,)
    plt.legend(loc="upper right")

    fig.suptitle("Training Curves on VOC")#add the title of the figure
    plt.show()


def test_plot(metrics,losses,forgets,metrics2,losses2,forgets2):
    from matplotlib.pyplot import MultipleLocator
    sns.set_style("darkgrid")
    fig, axs = plt.subplots(2,1)

    ax0 = axs[0]
    ax1 = axs[1]



    t = np.arange(1,len(metrics)+1,step=1)

    ax0.plot(t, metrics, color="red", label="Ours")
    #ax1.plot(t, losses, color="blue", label="1", )
    ax1.plot(t, forgets, color="orange", label="Ours",)

    ax0.plot(t, metrics2, color="red", label="ER",linestyle="--")
    #ax1.plot(t, losses2, color="blue", label="2", linestyle="--")
    ax1.plot(t, forgets2, color="orange", label="ER", linestyle="--")
    #ax1.plot(t, forgets, color="red", label="ss", linestyle="--")

    font1 = {'size': 14}
    font2= {"size": 11}

    # ax0.set_xlabel("Tasks", font1)
    ax0.set_ylabel("Macro-AUC", font2)

    # ax1.set_xlabel("Tasks", font1)
    # ax1.set_ylabel("Test error", )

    ax1.set_xlabel("Tasks", font1)
    ax1.set_ylabel("Forgetting",font2 )

    x_major_locator=MultipleLocator(1)
    ax0.xaxis.set_major_locator(x_major_locator)
    ax1.xaxis.set_major_locator(x_major_locator)


    ax0.set_title("Test performances on C-MSCOCO", font1)  # add the title of the figure
    ax0.legend()
    ax1.legend()
    plt.show()


if __name__ == '__main__':
    # voc
    file_name_ours = "train_logs/multilabel1set/voc/" \
                     "batch_learning-u_2-micro-auc-sgd/e40-s2222-b196/default.txt"
    file_name_er = "train_logs/multilabel1set/voc/" \
                     "batch_learning-u_1-micro-auc-sgd/e40-s2222-b196/default.txt"
    # coco
    # file_name_ours = "train_logs/multilabel1set/coco-8task/" \
    #                  "rmargin/_2000_0.9-ReWeightedMargin-macro-auc-sgd/e40-s2222-b240/default.txt"
    # file_name_er = "train_logs/multilabel1set/coco-8task/" \
    #                "er/_2000-BCE-macro-auc-sgd/e40-s2222-b96/default.txt"
    # nus-wide
    # file_name_ours = "train_logs/multilabel1set/nus-wide/" \
    #                  "batch_learning1e-4-u_2-micro-auc-sgd/e40-s2222-b392/default.txt"
    # file_name_er = "train_logs/multilabel1set/nus-wide/" \
    #                  "batch_learning-u_1-micro-auc-sgd/e40-s2222-b256/default.txt"
    # train plot

    metrcis,losses = read_training_log(file_name_ours)
    metrcis2, losses2 = read_training_log(file_name_er)

    train_plot(metrcis,losses,metrcis2, losses2)

    # test plot
    # m,l,f = read_test_log(file_name_ours)
    # m2, l2, f2 = read_test_log(file_name_er)
    # test_plot(m,l,f,m2, l2, f2)
