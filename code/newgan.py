import os, cv2, torch, random, itertools, time,datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage.interpolation import zoom
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train():
    base_lr = 0.001
    D_lr = 0.001
    ema_decay = 0.99
    num_classes = 2
    batch_size = 16
    labeled_slice = 100
    labeled_bs = int(batch_size/2)
    base_dir = './data/drive/'
    dataset = 'drive'
    image_size = (256,256)
    consistency = 0.1
    max_epoch = 200
    
    def create_model(ema=False):
        
        model = resnet34(num_classes)
        #model = UNet_plus(num_classes)
        #model = DeepLabv3 (num_classes)
        #model = PSPNet    (num_classes)
        #model = FCN(num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    pretrained_dict = torch.load('./resnet34.pth')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    summary(model, input_size=(3, image_size[0],image_size[1]), batch_size=1) 
    
    ema_model = create_model(ema=False)
    #ema_model.load_state_dict(pretrained_dict)
    ema_model.cuda()

    DAN = Discriminator(num_classes,1)
    DAN = DAN.cuda()
    
    db_train = vessel_BaseDataSets(base_dir+'train/', 'train.txt',image_size,dataset,transform=transforms.Compose([RandomGenerator()]))
    db_valid = vessel_BaseDataSets(base_dir+'test/', 'test.txt',image_size,dataset, transform=transforms.Compose([RandomGenerator()]))


    total_slices = len(db_train)
    print("Total images is: {}, labeled images is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    train_loader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=0, pin_memory=True)
    valid_loader = DataLoader(db_valid, batch_size=batch_size, shuffle=False,num_workers=0)
    print('train len:', len(train_loader))
    #optimizer = optim.SGD(model.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), betas=(0.9,0.99), lr=base_lr, weight_decay=0.0001)
    optimizer_T = optim.Adam(ema_model.parameters(), betas=(0.9,0.99), lr=0.0001, weight_decay=0.0001)
    DAN_optimizer = optim.Adam(DAN.parameters(), lr=D_lr, betas=(0.9, 0.99),weight_decay=0.0001)
    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    iter_num = 0
    max_iterations =  max_epoch * len(train_loader)
    distance = []
    valid_acc = []
    
    for epoch_num in range(max_epoch):
        train_acc = 0
        train_loss = 0
        test_acc =0
        Ttest_acc = 0
        start_time = time.time()
        
        print('Epoch: {} / {} '.format(epoch_num, max_epoch))
        for i_batch, sampled_batch in enumerate(train_loader):
            images, labels = sampled_batch['image'], sampled_batch['label']
            images, labels = images.cuda(), labels.cuda()
            unlabeled_images = images[labeled_bs:]
            
            model.train()
            #DAN.eval()
            for param in model.parameters():
                param.requires_grad = True
                
            for param in DAN.parameters():
                param.requires_grad = False
            
            noise = torch.clamp(torch.randn_like(unlabeled_images) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_images

            outputs = model(images)
            outputs_soft = torch.softmax(outputs, dim=1)
            supervised_loss = ce_loss(outputs[:labeled_bs],labels[:labeled_bs].long()) #+ dice_loss(outputs[:labeled_bs], labels[:labeled_bs], softmax=True)
            
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)
                ema_output_label = torch.max(ema_output_soft,1)[1]
            
            DAN_outputs = DAN(outputs_soft[labeled_bs:,1:2,:,:], ema_inputs)
            DAN_target = torch.tensor([0] * batch_size).cuda()
            DAN_target[:labeled_bs] = 1
            
            gan_loss = F.cross_entropy(DAN_outputs, DAN_target[:labeled_bs].long()) 
            

            if iter_num < 100:
                consistency_loss = 0.0
                unsupervised_loss = 0.0
                
            else:
                consistency_loss = torch.mean((outputs_soft[labeled_bs:]-ema_output_soft)**2) # all data
                unsupervised_loss = ce_loss(outputs[labeled_bs:],ema_output_label.long())
                
            
            consistency_weight = get_current_consistency_weight(consistency,epoch_num,max_epoch)
            loss = supervised_loss + consistency_weight* (consistency_loss + unsupervised_loss + gan_loss) 
            #loss = supervised_loss + consistency_weight* (unsupervised_loss + gan_loss) 
            
            prediction = torch.max(outputs[:labeled_bs],1)[1]
            train_correct = (prediction == labels[:labeled_bs]).float().mean().cpu().numpy()
            train_acc = train_acc + train_correct
            train_loss = train_loss + loss.detach().cpu().numpy()
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            #if iter_num > 300:    
            update_ema_variables(model, ema_model, ema_decay, iter_num)
            
            #model.eval()
            for param in model.parameters():
                param.requires_grad = False
            for param in DAN.parameters():
                param.requires_grad = True
            
            with torch.no_grad():
                outputs = model(images)
                outputs_soft = torch.softmax(outputs, dim=1)
                output_label = torch.max(outputs_soft,1)[1]
            
            #     changge      
            DAN_outputs1 = DAN(outputs_soft[:,1:2,:,:], images)
            DAN_target1 = torch.tensor([0] * batch_size).cuda()
            DAN_target1[0:labeled_bs] = 1
            D_loss = F.cross_entropy(DAN_outputs1, DAN_target1.long())
            
            DAN_optimizer.zero_grad()
            D_loss.backward()
            DAN_optimizer.step()
            
            lr = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            D_lr = D_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in DAN_optimizer.param_groups:
                param_group['lr'] = D_lr
            
            
            ema_output = ema_model(images)
            teacher_sup_loss = ce_loss(ema_output[:labeled_bs],labels[:labeled_bs].long()) #+ dice_loss(ema_output[:labeled_bs], labels[:labeled_bs], softmax=True)
            teacher_unsup_loss = ce_loss(ema_output[labeled_bs:],output_label[labeled_bs:].long())
            teacher_loss =  teacher_sup_loss + teacher_unsup_loss
            optimizer_T.zero_grad()
            teacher_loss.backward()
            optimizer_T.step()
            
            update_ema_variables(ema_model, model, ema_decay, iter_num)
            
            lr_T = 0.0001 * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_T
            
          
            iter_num = iter_num + 1
        
        model.eval()
        start_time = time.time()
        for i_batch, sampled_batch in enumerate(valid_loader):
            images, labels = sampled_batch['image'], sampled_batch['label']
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(images)
                outputs_soft = torch.softmax(outputs, dim=1)
                prediction = torch.max(outputs,1)[1]
                
                test_correct = (prediction == labels).float().mean().cpu().numpy()
                test_acc = test_acc + test_correct
        
        #valid_acc.append(round(test_acc/len(valid_loader),2))       
        print('train_loss: ',train_loss/(labeled_slice/labeled_bs),' train_acc: ',train_acc/(labeled_slice/labeled_bs),'test_acc: ',test_acc/len(valid_loader)) 
        model.train()
        if epoch_num>=50 and epoch_num%3 == 0:
            torch.save(model.state_dict(), './new/our_student_drive_'+str(epoch_num)+'.pth')  
            torch.save(ema_model.state_dict(), './new/our_teacher_drive_'+str(epoch_num)+'.pth') 
train()