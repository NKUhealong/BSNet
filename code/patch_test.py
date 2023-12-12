from models import *
from dataset import *
from utils import *
from efficientNet import *
from torchvision import transforms
import copy
import torch

def extract_ordered_overlap(image,patch_size,s):
    
    image_h = image.shape[2]  
    image_w = image.shape[3] 
    N_patches_img = ((image_h-patch_size)//s+1)*((image_w-patch_size)//s+1) 
    
    patches = torch.empty((N_patches_img,3,patch_size,patch_size))
    #print(patches.shape)
    iter_tot = 0   
    for h in range((image_h-patch_size)//s+1):
        for w in range((image_w-patch_size)//s+1):
            patch = image[0,:,h*s:(h*s)+patch_size,w*s:(w*s)+patch_size]
            patches[iter_tot]=patch
            iter_tot =iter_tot+1      
    return patches

def recompone_overlap(preds, img_h, img_w, s):
    #print(preds.shape)
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    N_patches_h = (img_h-patch_h)//s+1
    N_patches_w = (img_w-patch_w)//s+1
    N_patches_img = N_patches_h * N_patches_w
    
    full_prob = torch.zeros((img_h,img_w)).cuda()
    full_sum =  torch.zeros((img_h,img_w)).cuda()
    k = 0
   
    for h in range((img_h-patch_h)//s+1):
        for w in range((img_w-patch_w)//s+1):
            
            full_prob[h*s:(h*s)+patch_h,w*s:(w*s)+patch_w]+=preds[k]
            full_sum [h*s:(h*s)+patch_h,w*s:(w*s)+patch_w]+=1
            k+=1
    
    final_avg = full_prob/full_sum
    
    return final_avg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
num_classes = 2
batch_size = 1
image_size = (512,512)
save_dir='./result/'
patch_size = 256
stride = 128
MT = 1
base_dir = './data/drive/test/'
dataset = 'polyp'

db_val = testBaseDataSets(base_dir, 'test.txt',image_size,dataset,transform=transforms.Compose([RandomGenerator()]))
valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False,num_workers=0)


if MT:
    
    T_model_name = 'UAMT_teacher_drive_'  # our_resnet_teacher_skin_
    S_model_name = 'UAMT_student_drive_'  # our_resnet_student_skin_
    
    T_model = resnet34   (num_classes)
    S_model = copy.deepcopy(T_model)
    print(id(T_model),id(S_model))
    for k in range(99,303,3): 
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
                
                ####add######
                patches = extract_ordered_overlap(images,patch_size,stride)
                #print(patches.shape)
                predictions_T = T_model(patches.cuda())
                pred_T = predictions_T[:,1,:,:]
                pred_T = recompone_overlap(pred_T, 512, 512, stride)
                #print(pred_T.shape)
                
                evaluator_T.update(pred_T, labels[0,:,:].float())

                predictions_S = S_model(patches.cuda())
                pred_S = predictions_S[:,1,:,:]
                pred_S = recompone_overlap(pred_S, 512, 512, stride)
                evaluator_S.update(pred_S, labels[0,:,:].float())

        evaluator_T.show()
        evaluator_S.show()
else:
    
    model_name='URPC_drive_'     #URPC_drive_   
    model = resnet34   (num_classes)
     
    for k in range(99,300,3): 
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
                patches = extract_ordered_overlap(images,patch_size,stride)
                #print(patches.shape)
                #outputs_tanh, predictions = model(patches.cuda())
                predictions,_,_,_ = model(patches.cuda())
                #predictions = model(patches.cuda())
                #print(predictions.shape)
                pred = predictions[:,1,:,:]
                pred = recompone_overlap(pred, 512, 512, stride)
                
                #print(pred.shape)
                evaluator.update(pred, labels[0,:,:].float())

                for i in range(batch_size):
                    labels = labels.cpu().numpy()
                    label = (labels[i]*255)
                    pred = pred.cpu().numpy()
                    #total_img = np.concatenate((label,pred[:,:]*255),axis=1)
                    cv2.imwrite(save_dir+'GT_Pre'+str(j)+'.jpg',pred[:,:]*255)
                    j=j+1
        evaluator.show()

    