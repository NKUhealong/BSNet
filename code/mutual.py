import os
import cv2
import torch
import time
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms

from models import *
from dataset import *
from torchsummary import summary
from utils import *
from efficientNet import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train():
    base_lr = 0.001
    T_lr = 0.0001
    ema_decay = 0.99
    num_classes = 2
    batch_size = 14
    labeled_slice = 100
    labeled_bs = int(batch_size/2)
    base_dir = './data/polyp/'
    dataset = 'polyp'
    image_size = (512,512)
    consistency = 0.1
    max_epoch = 200
    model_arch = 'mobilev2'
    
    weights_dict = {'resnet':torch.load('./resnet34.pth'),       'vgg16': torch.load('./vgg16_bn.pth'),
                    'mobilev2':torch.load('./mobilenet_v2.pth'),   'efficient': torch.load('./efficientnet_b0.pth')}
    
    def create_model(ema=False):
        models_dict = {'resnet':resnet34(pretrained = False,num_classes = num_classes),   'vgg16': VGG16(num_classes,True),
                   'mobilev2':MobileNetV2(num_classes = num_classes), 'efficient': EfficientNet.from_name('efficientnet-b0',num_classes = num_classes) }
        model = models_dict[model_arch]
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    pretrained_dict = weights_dict[model_arch]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    summary(model, input_size=(3, 512,512), batch_size=1)  
    
    ema_model = create_model(ema=True)
    ema_model.load_state_dict(model_dict)
    ema_model.cuda()
    print(id(model),id(ema_model))
    
    db_train = vessel_BaseDataSets(base_dir+'train/', 'train.txt',image_size,dataset,transform=transforms.Compose([RandomGenerator()]))
    db_valid = vessel_BaseDataSets(base_dir+'test/', 'test.txt',image_size,dataset, transform=transforms.Compose([RandomGenerator()]))

    total_slices = len(db_train)
    print("Total images is: {}, labeled images is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    train_loader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=0, pin_memory=True)
    valid_loader = DataLoader(db_valid, batch_size=batch_size, shuffle=False,num_workers=0)
    print('Iteration number of each epoch:', len(train_loader))
    
    optimizer = optim.Adam(model.parameters(), betas=(0.9,0.99), lr=base_lr, weight_decay=0.0001)
    optimizer_T = optim.Adam(ema_model.parameters(), betas=(0.9,0.99), lr=T_lr, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_iterations =  max_epoch * len(train_loader)
    
    for epoch_num in range(max_epoch):
        train_acc = 0
        train_loss = 0
        test_acc =0

        print('Epoch: {} / {} '.format(epoch_num, max_epoch))
        for i_batch, sampled_batch in enumerate(train_loader):
            images, labels = sampled_batch['image'], sampled_batch['label']
            images, labels = images.cuda(), labels.cuda()
            unlabeled_images = images[labeled_bs:]
            ema_inputs = unlabeled_images
            
            outputs = model(images)
            outputs_soft = torch.softmax(outputs, dim=1)
            supervised_loss = ce_loss(outputs[:labeled_bs],labels[:labeled_bs].long())
            
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)
                ema_output_label = torch.max(ema_output_soft,1)[1]
                #   add reliable masks

            if iter_num < 100:
                consistency_loss = 0.0
                unsupervised_loss = 0.0
                
            else:
                consistency_loss = 0.0
                #consistency_loss = torch.mean((outputs_soft[labeled_bs:]-ema_output_soft)**2)
                unsupervised_loss = ce_loss(outputs[labeled_bs:],ema_output_label.long())
                
            
            consistency_weight = get_current_consistency_weight(consistency,epoch_num,max_epoch)
            loss = supervised_loss + consistency_weight* (consistency_loss + unsupervised_loss) 
            
            prediction = torch.max(outputs[:labeled_bs],1)[1]
            train_correct = (prediction == labels[:labeled_bs]).float().mean().cpu().numpy()
            train_acc = train_acc + train_correct
            train_loss = train_loss + loss.detach().cpu().numpy()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iter_num%3 == 0:
                update_ema_variables(model, ema_model, ema_decay, iter_num)
            
            lr = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            ###########################
            with torch.no_grad():
                outputs = model(images)
                outputs_soft = torch.softmax(outputs, dim=1)
                output_label = torch.max(outputs_soft,1)[1]   
                
            ema_output = ema_model(images)    
            #teacher_sup_loss = ce_loss(ema_output[:labeled_bs],labels[:labeled_bs].long())
            if iter_num < 100:
                teacher_unsup_loss = 0.0
            else:
                teacher_unsup_loss = ce_loss(ema_output[labeled_bs:],output_label[labeled_bs:].long())     
                teacher_loss =  consistency_weight* teacher_unsup_loss
                optimizer_T.zero_grad()
                teacher_loss.backward()
                optimizer_T.step()
            lr_T = T_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_T
                
            if iter_num > 100 and iter_num%3 == 0:
                update_ema_variables(ema_model, model, ema_decay, iter_num)
            #######################
            
            iter_num = iter_num + 1
            
        ##  test
        model.eval()
        for i_batch, sampled_batch in enumerate(valid_loader):
            images, labels = sampled_batch['image'], sampled_batch['label']
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(images)
                outputs_soft = torch.softmax(outputs, dim=1)
                prediction = torch.max(outputs,1)[1]
                test_correct = (prediction == labels).float().mean().cpu().numpy()
                test_acc = test_acc + test_correct
        print('train_loss: ',train_loss/(labeled_slice/labeled_bs),' train_acc: ',train_acc/(labeled_slice/labeled_bs),'test_acc: ',test_acc/len(valid_loader)) 
        model.train()
        if epoch_num>=50 and epoch_num%3 == 0:
            torch.save(model.state_dict(), './new/student_mutual_'+str(epoch_num)+'.pth')  
            torch.save(ema_model.state_dict(), './new/teacher_mutual_'+str(epoch_num)+'.pth') 

train()