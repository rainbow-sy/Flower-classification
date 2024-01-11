import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
# from pre_dataLoad import load_data
from DataLoad import load_data
# from Modified_Base_Model import convnext_tiny as create_model
from final_model import convnext_tiny as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate,test,plot_fig
from sklearn.model_selection import train_test_split
import time
class BrainTumorDataset(Dataset):
    def __init__(self, images, labels):
        # images
        self.X = images
        # labels
        self.y = labels
        # Transformation for converting original image array to an image and then convert it to a tensor
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = self.transform(self.X[idx])
        target = self.y[idx]
        return data, target

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    img_size = 224
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    X,Y=load_data()
    print('所有数据',len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        shuffle=True)  # 70% training, 30% testing
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
    print(len(X_train))
    print(len(X_valid))
    print(len(X_test))
    train_set = BrainTumorDataset(X_train, y_train)
    valid_set = BrainTumorDataset(X_valid, y_valid)
    test_set = BrainTumorDataset(X_test, y_test)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    test_loader = DataLoader(test_set, batch_size=batch_size*4, shuffle=True, pin_memory=True, num_workers=nw)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k] 
        #将strict设置为False，可以在两个模型不同的情况下，仅加载相同键值部分
        print(model.load_state_dict(weights_dict, strict=False))
    # model.load_state_dict(torch.load(args.weights), strict=False)


    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=5)
    lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=2)   #上一次的设置为factor=0.1, patience=3

    best_acc = 0.
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    all_time = []
    if os.path.exists('./模型结果.txt'):
        os.remove('./模型结果.txt')
    filename = './模型结果.txt'
    for epoch in range(args.epochs):
        # train
        t1 = time.time()
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler1=lr_scheduler
                                                ,lr_scheduler2=lr_scheduler2)
        
        t2 = time.time()
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        lr_scheduler2.step(val_loss)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)
        all_time.append(t2-t1)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "./weights/best_model.pth")
            best_acc = val_acc
    with open(filename, 'a+') as f:
        f.write('训练集损失：' + str(all_train_loss) + '\n')
        f.write('训练集准确率：' + str(all_train_acc) + '\n')
        f.write('验证集损失：' + str(all_val_loss) + '\n')
        f.write('验证集准确率：' + str(all_val_acc) + '\n')
        f.write('训练时间：' + str(all_time) + '\n')
    t3 = time.time()
    test_acc,microscore_f1,macroscore_f1, macro_precision, macro_recall=test(test_loader, device=device)
    t4= time.time()
    with open(filename, 'a+') as f:
        f.write('测试集准确率：{} .'.format(test_acc)+ '\n')
        f.write('测试集micro_f1：{} .'.format(microscore_f1)+ '\n')
        f.write('测试集macro_f1：{} .'.format(macroscore_f1)+ '\n')
        f.write('测试集macro_precision：{} .'.format(macro_precision)+ '\n')
        f.write('测试集macro_recall：{} .'.format(macro_recall)+ '\n')
        f.write('推理时间：' + str(t4-t3) + '\n')
    #-------------------------绘图------------------------#  
    plot_fig(all_train_loss,all_val_loss)
    plot_fig(all_train_acc,all_val_acc,metrics='Accuracy')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='./convnext_tiny_1k_224_ema.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
