import os
import sys
import json
import pickle
import random
import math
from final_model import convnext_tiny as create_model
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def read_split_data(root:str, val_rate:float=0.2):
    random.seed(0)
    assert os.path.exists(root),"dataset root {} does not exist.".format(root)
    flower_class=[cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root,cla))]
    flower_class.sort()
    class_indices=dict((k,v) for v,k in enumerate(flower_class))
    json_str=json.dumps((val,key) for key,val in class_indices.items())
    with open("class_indices.json",'w') as json_file:
        json_file.write(json_str)


def plot_data_loader_image(data_loader):
    batch_size=data_loader.batch_size
    plot_num=min(batch_size,4)

    json_path='./class_indices.json'
    assert os.path.exists(json_path),json_path+"does not exist."
    json_file=open(json_path,'r')
    class_indices=json.load(json_file)

    for data in data_loader:
        images,labels=data
        for i in range(plot_num):
            img=images[i].numpy().transpose(1,2,0)
            img=(img*[0.229,0.224,0.225]+[0.485,0.456,0.406])*255
            label=labels[i].item()
            plt.subplot(1,plot_num,i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xsticks([])
            plt.ysticks([])
            plt.imshow(img.astype("uint8"))
        plt.show()

def write_pickle(list_info:list, file_name:str):
    with open(file_name,'wb') as f:
        info_list=pickle.dump(list_info,f)
        return info_list

def train_one_epoch(model,optimizer,data_loader,device,epoch,lr_scheduler1,lr_scheduler2):
    model.train()
    loss_function=torch.nn.CrossEntropyLoss()
    accu_loss=torch.zeros(1).to(device)
    accu_num=torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num=0
    data_loader=tqdm(data_loader,file=sys.stdout)
    for step,data in enumerate(data_loader):
        images,labels=data
        sample_num+=images.shape[0]

        pred=model(images.to(device))
        pred_classes=torch.max(pred,dim=1)[1]
        accu_num+=torch.eq(pred_classes,labels.to(device)).sum()

        loss=loss_function(pred,labels.to(device))
        loss.backward()
        accu_loss+=loss.detach()

        data_loader.desc="[train epoch{}] loss: {:.3f}, acc:{:.3f}, lr:{:.5f}".format(
            epoch,
            accu_loss.item()/(step+1),
            accu_num.item()/sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler1.step()
    return accu_loss.item()/(step+1), accu_num.item()/sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function=torch.nn.CrossEntropyLoss

    model.eval()
    accu_num=torch.zeros(1).to(device)
    accu_loss=torch.zeros(1).to(device)

    sample_num=0
    data_loader=tqdm(data_loader,file=sys.stdout)

    for step,data in enumerate(data_loader):
        images, labels=data
        sample_num_=images.shape[0]

        pred=model(images.to(device))
        pred_classes=torch.max(pred,dim=1)[1]
        accu_num_=torch.eq(pred_classes, labels.to(device)).sum()

        loss=loss_function(pred,labels.to(device))
        accu_loss+=loss
        data_loader.desc="[valid epoch {}] loss:{:.3f}, acc:{:.3f}".format(
            epoch,
            accu_loss.item()/(step+1),
            accu_num.item()/sample_num
        )
    return accu_loss.item()/(step+1),accu_num.item()/sample_num

def test(data_loader,device):
    model=create_model(num_classes=4).to(device)
    model_weight_path='./weights/best_model.pth'
    model.load_state_dict(torch.load(model_weight_path,map_location=device))
    model.eval()
    accu_num=torch.zeros(1).to(device)
    sample_num=0
    data_loader=tqdm(data_loader,file=sys.stdout)
    result_list=[]
    labels_list=[]
    with torch.no_gtrad():
        for step,data in enumerate(data_loader):
            images,labels=data
            sample_num+=images.shape[0]
            pred=model(images.to(device))
            pred_classes=torch.max(pred,dim=1)[1]
            result_list.append(pred_classes)
            labels_list.append(labels)
            accu_num+=torch.eq(pred_classes, labels.to(device)).sum()
        result_final=torch.cat(result_list,dim=0).cpu()
        labels_final=torch.cat(labels_list,dim=0).cpu()
    micro_f1=f1_score(labels_final,result_final,average='micro')
    macro_f1=f1_score(labels_final,result_final,average='macro')
    Macro_precision=precision_score(labels_final,result_final,average='macro')
    Macro_recall=recall_score(labels_final,result_final,average='macro')
    cm=confusion_matrix(labels_final,result_final)
    conf_matrix=pd.DataFrame(cm,index=['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia'],columns=['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia'])
    fig,ax=plt.subplots(fig_size=(10,10))
    ax.get_yaxis().get_major_formatter().set_scientific(True)
    sns.heatmap(conf_matrix,annot=True,annot_kws={"size":19},cmap="Blues",fmt=".20g")
    plt.ylabel("True label", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig()
    plt.show()
    return accu_num.item()/sample_num,micro_f1,macro_f1,Macro_precision,Macro_recall

def create_lr_scheduler(optimizer,
                        num_step:int,
                        epochs:int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step>0 and epochs>0
    if warmup is False:
        warmup_epochs=0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha=float(x)/(warmup_epochs * num_step)
            return warmup_factor*(1-alpha)+alpha
        else:
            current_step=(x-warmup_epochs*num_step)
            cosine_steps=(epochs-warmup_epochs)*num_step
            return ((1+math.cos(current_step*math.pi)/cosine_steps)/2) * (1-end_factor)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def get_params_groups(model: torch.nn.Module, weight_decay: float=1e-5):
    parameter_group_vars={"decay":{"params":[],"weight_decay":weight_decay},
                          "no_decay":{"params:[],:weight_decay":0.}}
    parameter_group_names={"decay":{"params":[],"weight_decay":weight_decay},
                          "no_decay":{"params:[],:weight_decay":0.}}

    for name,param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape)==1 or name.endswith(".bias"):
            group_name="no_decay"
        else:
            group_name="decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s"%json.dumps(parameter_group_names,indent=2))
    return list(parameter_group_vars.values())

def plot_fig(train_loss,valid_loss,metrics='Loss'):
    plt.figure(figsize=(8,6))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.legend(["Training"+metrics,"Validation "+metrics])
    plt.xlabel("Epochs",fontsize=16)
    plt.ylabel(metrics,fontsize=16)
    plt.title(metrics+"Curves",fontsize=16)
    plt.savefig(r"./model_"+metrics+'.png')
    plt.show()