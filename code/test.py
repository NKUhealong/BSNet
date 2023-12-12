from models import *
from dataset import *
from utils import *
from efficientNet import *
from torchvision import transforms
import copy

import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch.distributions.uniform import Uniform


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
num_classes = 2
batch_size = 1
image_size = (512,512)
save_dir='./result/'

MT = 0
base_dir = './data/polyp/test/'
dataset = 'polyp'

db_val = testBaseDataSets(base_dir, 'test.txt',image_size,dataset,transform=transforms.Compose([RandomGenerator()]))
valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False,num_workers=0)


if MT:
    
    T_model_name = 'UAMT_teacher_polyp_'  # our_teacher_polyp_
    S_model_name = 'UAMT_student_polyp_'  # our_student_polyp_
    
    T_model = resnet34(num_classes)
    #T_model = UNet_plus(num_classes)
    #T_model = DeepLabv3(num_classes)
    #T_model = PSPNet (num_classes)
    #T_model = FCN(num_classes)
    S_model = copy.deepcopy(T_model)
    print(id(T_model),id(S_model))
    for k in range(99,250,3): 
        print('./new/'+T_model_name+str(k)+'.pth')
        T_model.load_state_dict(torch.load('./new/'+T_model_name+str(k)+'.pth'))
        T_model.cuda()
        T_model.eval()

        S_model.load_state_dict(torch.load('./new/'+S_model_name+str(k)+'.pth'))
        S_model.cuda()
        S_model.eval()

        j = 0
        evaluator_T= Evaluator()
        evaluator_S= Evaluator()
        with torch.no_grad():
            for sampled_batch in valloader:
                images, labels = sampled_batch['image'], sampled_batch['label']
                images, labels = images.cuda(),labels.cuda()

                predictions_T = T_model(images)
                pred_T = predictions_T[0,1,:,:]
                evaluator_T.update(pred_T, labels[0,:,:].float())

                predictions_S = S_model(images)
                pred_S = predictions_S[0,1,:,:]
                evaluator_S.update(pred_S, labels[0,:,:].float())

        evaluator_T.show()
        evaluator_S.show()
else:
    
    model_name='URPC_polyp_'   #
    #model = resnet34   (num_classes)
    #model = UNet_plus (num_classes)
    #model = DeepLabv3 (num_classes)
    #model = PSPNet    (num_classes)
    #model = FCN(num_classes)
    #model = DTC(num_classes)
    model = URPC(num_classes)
     
    for k in range(21,230,3): 
        print('./new/'+model_name+str(k)+'.pth')
        model.load_state_dict(torch.load('./new/'+model_name+str(k)+'.pth'))
        model.cuda()
        model.eval()
        j = 0
        evaluator = Evaluator()
        with torch.no_grad():
            for sampled_batch in valloader:
                images, labels = sampled_batch['image'], sampled_batch['label']
                images, labels = images.cuda(),labels.cuda()
                #outputs_tanh, predictions = model(images)
                predictions,_,_,_ = model(images)
                #predictions = model(images)
                pred = predictions[0,1,:,:]
                evaluator.update(pred, labels[0,:,:].float())

                for i in range(batch_size):
                    labels = labels.cpu().numpy()
                    label = (labels[i]*255)
                    pred = pred.cpu().numpy()
                    #total_img = np.concatenate((label,pred[:,:]*255),axis=1)
                    cv2.imwrite(save_dir+'Pre'+str(j)+'.jpg',pred[:,:]*255)
                    j=j+1

        evaluator.show()

    