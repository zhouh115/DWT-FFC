import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import os
import time
import re
from torchvision import transforms
from utils_test import to_psnr, to_ssim_skimage    #少一个to_rmse
from tensorboardX import SummaryWriter

from pytorch_msssim import msssim
from perceptual import LossNetwork

from test_dataset_for_testing import dehaze_test_dataset
from model_convnext import fusion_net

#os.chdir("UNet-FFC-dehazing")

parser = argparse.ArgumentParser(description='Dehaze')
parser.add_argument('--test_dir', type=str, default='./NTIRE2023/NTIRE2023_Test_Hazy/')
parser.add_argument('--output_dir', type=str, default='New Checkpoints/3-dwt-ffc-original-convnext-original-lr-new-dataset/finetune_3849_25-17-25/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
args = parser.parse_args()
output_dir =args.output_dir
if not os.path.exists(output_dir + '/'):
    os.makedirs(output_dir + '/')
test_dir = args.test_dir
test_batch_size = args.test_batch_size

test_dataset = dehaze_test_dataset(test_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(device_ids[0])
print(device)
# --- Define the network --- #
MyEnsembleNet= fusion_net()
#writer = SummaryWriter()
# --- Multi-GPU --- #
MyEnsembleNet = MyEnsembleNet.to(device)
number=0
# for i in os.listdir(r'E:/GPU0/5-revised-dwt-ffc-convnext-refinement-120/lr-original/1/2/'):
#   MyEnsembleNet.load_state_dict(torch.load(os.path.join(r'E:/GPU0/5-revised-dwt-ffc-convnext-refinement-120/lr-original/1/2/', i)))
#   number=number+1
for i in os.listdir('New Checkpoints/3-dwt-ffc-original-convnext-original-lr-new-dataset/checkpoints/finetune_3849/25-17based/1/'):
  MyEnsembleNet.load_state_dict(torch.load(os.path.join('New Checkpoints/3-dwt-ffc-original-convnext-original-lr-new-dataset/checkpoints/finetune_3849/25-17based/1/', i)))
  number=number+1

  print('we are testing on'+str(i))
  # --- Test --- #
  with torch.no_grad():
    psnr_list = []
    ssim_list = []
    rmse_list = []
    MyEnsembleNet.eval()
    start_time = time.time()
    # with open('New Checkpoints/3-dwt-ffc-original-convnext-original-lr-new-dataset/test4-larger-than-21.6.txt', 'a') as f:
    #   f.write("epoch:" + str(i) + ";")
    for batch_idx, (hazy_up_left, hazy_up_middle, hazy_up_right, hazy_middle_left, hazy_middle_middle, hazy_middle_right, hazy_down_left, hazy_down_middle, hazy_down_right,name) in enumerate(test_loader):
      #print(hazy_up_left.shape)
      #clean = clean.to(device)
      hazy_up_left = hazy_up_left.to(device)
      hazy_up_middle =hazy_up_middle.to(device)
      hazy_up_right = hazy_up_right.to(device)                
      hazy_middle_left =hazy_middle_left.to(device)
      hazy_middle_middle = hazy_middle_middle.to(device)
      hazy_middle_right =hazy_middle_right.to(device)

      hazy_down_left = hazy_down_left.to(device)
      hazy_down_middle =hazy_down_middle.to(device)
      hazy_down_right = hazy_down_right.to(device)

      frame_out_up_left = MyEnsembleNet(hazy_up_left)
      frame_out_middle_left = MyEnsembleNet(hazy_middle_left)
      frame_out_down_left = MyEnsembleNet(hazy_down_left)

      frame_out_up_middle = MyEnsembleNet(hazy_up_middle)
      frame_out_middle_middle = MyEnsembleNet(hazy_middle_middle)
      frame_out_down_middle = MyEnsembleNet(hazy_down_middle)

      frame_out_up_right = MyEnsembleNet(hazy_up_right)
      frame_out_middle_right = MyEnsembleNet(hazy_middle_right)
      frame_out_down_right = MyEnsembleNet(hazy_down_right)

      frame_out_up_left=frame_out_up_left.to(device)
      frame_out_middle_left =frame_out_middle_left .to(device)
      frame_out_down_left=frame_out_down_left.to(device)
      frame_out_up_middle=frame_out_up_middle.to(device)
      frame_out_middle_middle=frame_out_middle_middle.to(device)
      frame_out_down_middle=frame_out_down_middle.to(device)
      frame_out_up_right=frame_out_up_right.to(device)
      frame_out_middle_right=frame_out_middle_right.to(device)
      frame_out_down_right=frame_out_down_right.to(device)

        

      if frame_out_up_left.shape[2]==1600:
        frame_out_up_left_middle=(frame_out_up_left[:,:,:,1800:2432]+frame_out_up_middle[:,:,:,0:632])/2
        frame_out_up_middle_right=(frame_out_up_middle[:,:,:,1768:2432]+frame_out_up_right[:,:,:,0:664])/2

        frame_out_middle_left_middle=(frame_out_middle_left[:,:,:,1800:2432]+frame_out_middle_middle[:,:,:,0:632])/2
        frame_out_middle_middle_right=(frame_out_middle_middle[:,:,:,1768:2432]+frame_out_middle_right[:,:,:,0:664])/2

        frame_out_down_left_middle=(frame_out_down_left[:,:,:,1800:2432]+frame_out_down_middle[:,:,:,0:632])/2
        frame_out_down_middle_right=(frame_out_down_middle[:,:,:,1768:2432]+frame_out_down_right[:,:,:,0:664])/2

          

        frame_out_left_up_middle=(frame_out_up_left[:,:,1200:1600,0:1800]+frame_out_middle_left[:,:,0:400,0:1800])/2
        frame_out_left_middle_down=(frame_out_middle_left[:,:,1200:1600,0:1800]+frame_out_down_left[:,:,0:400,0:1800])/2

        frame_out_left = (torch.cat([frame_out_up_left[:, :, 0:1200, 0:1800].permute(0, 2, 3, 1),frame_out_left_up_middle.permute(0, 2, 3, 1), frame_out_middle_left[:, :, 400:1200, 0:1800].permute(0, 2, 3, 1), frame_out_left_middle_down.permute(0, 2, 3, 1), frame_out_down_left[:, :, 400:, 0:1800].permute(0, 2, 3, 1)],1))
          
          
        frame_out_leftmiddle_up_middle=(frame_out_up_left_middle[:,:,1200:1600,:]+frame_out_middle_left_middle[:,:,0:400,:])/2
        frame_out_leftmiddle_middle_down=(frame_out_middle_left_middle[:,:,1200:1600,:]+frame_out_down_left_middle[:,:,0:400,:])/2


        frame_out_leftmiddle = (torch.cat([frame_out_up_left_middle[:, :, 0:1200, :].permute(0, 2, 3, 1),frame_out_leftmiddle_up_middle.permute(0, 2, 3, 1), frame_out_middle_left_middle[:, :, 400:1200, :].permute(0, 2, 3, 1), frame_out_leftmiddle_middle_down.permute(0, 2, 3, 1), frame_out_down_left_middle[:, :, 400:, :].permute(0, 2, 3, 1)],1))
          
          
        frame_out_middle_up_middle=(frame_out_up_middle[:,:,1200:1600,632:1768]+frame_out_middle_middle[:,:,0:400,632:1768])/2
        frame_out_middle_middle_down=(frame_out_middle_middle[:,:,1200:1600,632:1768]+frame_out_down_middle[:,:,0:400,632:1768])/2
          
        frame_out_middle = (torch.cat([frame_out_up_middle[:, :, 0:1200, 632:1768].permute(0, 2, 3, 1),frame_out_middle_up_middle.permute(0, 2, 3, 1), frame_out_middle_middle[:, :, 400:1200, 632:1768].permute(0, 2, 3, 1), frame_out_middle_middle_down.permute(0, 2, 3, 1), frame_out_down_middle[:, :, 400:, 632:1768].permute(0, 2, 3, 1)],1))
          
        frame_out_middleright_up_middle=(frame_out_up_middle_right[:,:,1200:1600,:]+frame_out_middle_middle_right[:,:,0:400,:])/2
        frame_out_middleright_middle_down=(frame_out_middle_middle_right[:,:,1200:1600,:]+frame_out_down_middle_right[:,:,0:400,:])/2

        frame_out_middleright = (torch.cat([frame_out_up_middle_right[:, :, 0:1200, :].permute(0, 2, 3, 1),frame_out_middleright_up_middle.permute(0, 2, 3, 1), frame_out_middle_middle_right[:, :, 400:1200, :].permute(0, 2, 3, 1), frame_out_middleright_middle_down.permute(0, 2, 3, 1), frame_out_down_middle_right[:, :, 400:, :].permute(0, 2, 3, 1)],1))

          



        frame_out_right_up_middle=(frame_out_up_right[:,:,1200:1600,664:]+frame_out_middle_right[:,:,0:400,664:])/2
        frame_out_right_middle_down=(frame_out_middle_right[:,:,1200:1600,664:]+frame_out_down_right[:,:,0:400,664:])/2

        frame_out_right = (torch.cat([frame_out_up_right[:, :, 0:1200, 664:].permute(0, 2, 3, 1),frame_out_right_up_middle.permute(0, 2, 3, 1), frame_out_middle_right[:, :, 400:1200, 664:].permute(0, 2, 3, 1), frame_out_right_middle_down.permute(0, 2, 3, 1), frame_out_down_right[:, :, 400:, 664:].permute(0, 2, 3, 1)],1))


          
          
      if frame_out_up_left.shape[2]==2432:
        frame_out_up_left_middle=(frame_out_up_left[:,:,:,1200:1600]+frame_out_up_middle[:,:,:,0:400])/2
        frame_out_up_middle_right=(frame_out_up_middle[:,:,:,1200:1600]+frame_out_up_right[:,:,:,0:400])/2

        frame_out_middle_left_middle=(frame_out_middle_left[:,:,:,1200:1600]+frame_out_middle_middle[:,:,:,0:400])/2
        frame_out_middle_middle_right=(frame_out_middle_middle[:,:,:,1200:1600]+frame_out_middle_right[:,:,:,0:400])/2

        frame_out_down_left_middle=(frame_out_down_left[:,:,:,1200:1600]+frame_out_down_middle[:,:,:,0:400])/2
        frame_out_down_middle_right=(frame_out_down_middle[:,:,:,1200:1600]+frame_out_down_right[:,:,:,0:400])/2

          

        frame_out_left_up_middle=(frame_out_up_left[:,:,1800:2432,0:1200]+frame_out_middle_left[:,:,0:632,0:1200])/2
        frame_out_left_middle_down=(frame_out_middle_left[:,:,1768:2432,0:1200]+frame_out_down_left[:,:,0:664,0:1200])/2

        frame_out_left = (torch.cat([frame_out_up_left[:, :, 0:1800, 0:1200].permute(0, 2, 3, 1),frame_out_left_up_middle.permute(0, 2, 3, 1), frame_out_middle_left[:, :, 632:1768, 0:1200].permute(0, 2, 3, 1), frame_out_left_middle_down.permute(0, 2, 3, 1), frame_out_down_left[:, :, 664:, 0:1200].permute(0, 2, 3, 1)],1))
          
          
        frame_out_leftmiddle_up_middle=(frame_out_up_left_middle[:,:,1800:2432,:]+frame_out_middle_left_middle[:,:,0:632,:])/2
        frame_out_leftmiddle_middle_down=(frame_out_middle_left_middle[:,:,1768:2432,:]+frame_out_down_left_middle[:,:,0:664,:])/2


        frame_out_leftmiddle = (torch.cat([frame_out_up_left_middle[:, :, 0:1800, :].permute(0, 2, 3, 1),frame_out_leftmiddle_up_middle.permute(0, 2, 3, 1), frame_out_middle_left_middle[:, :, 632:1768, :].permute(0, 2, 3, 1), frame_out_leftmiddle_middle_down.permute(0, 2, 3, 1), frame_out_down_left_middle[:, :, 664:, :].permute(0, 2, 3, 1)],1))
          
          
        frame_out_middle_up_middle=(frame_out_up_middle[:,:,1800:2432,400:1200]+frame_out_middle_middle[:,:,0:632,400:1200])/2
        frame_out_middle_middle_down=(frame_out_middle_middle[:,:,1768:2432,400:1200]+frame_out_down_middle[:,:,0:664,400:1200])/2
          
        frame_out_middle = (torch.cat([frame_out_up_middle[:, :, 0:1800, 400:1200].permute(0, 2, 3, 1),frame_out_middle_up_middle.permute(0, 2, 3, 1), frame_out_middle_middle[:, :, 632:1768, 400:1200].permute(0, 2, 3, 1), frame_out_middle_middle_down.permute(0, 2, 3, 1), frame_out_down_middle[:, :, 664:, 400:1200].permute(0, 2, 3, 1)],1))
          
          
          
        frame_out_middleright_up_middle=(frame_out_up_middle_right[:,:,1800:2432,:]+frame_out_middle_middle_right[:,:,0:632,:])/2
        frame_out_middleright_middle_down=(frame_out_middle_middle_right[:,:,1768:2432,:]+frame_out_down_middle_right[:,:,0:664,:])/2

        frame_out_middleright = (torch.cat([frame_out_up_middle_right[:, :, 0:1800, :].permute(0, 2, 3, 1),frame_out_middleright_up_middle.permute(0, 2, 3, 1), frame_out_middle_middle_right[:, :, 632:1768, :].permute(0, 2, 3, 1), frame_out_middleright_middle_down.permute(0, 2, 3, 1), frame_out_down_middle_right[:, :, 664:, :].permute(0, 2, 3, 1)],1))

          


        frame_out_right_up_middle=(frame_out_up_right[:,:,1800:2432,400:]+frame_out_middle_right[:,:,0:632,400:])/2
        frame_out_right_middle_down=(frame_out_middle_right[:,:,1768:2432,400:]+frame_out_down_right[:,:,0:664,400:])/2

        frame_out_right = (torch.cat([frame_out_up_right[:, :, 0:1800, 400:].permute(0, 2, 3, 1),frame_out_right_up_middle.permute(0, 2, 3, 1), frame_out_middle_right[:, :, 632:1768, 400:].permute(0, 2, 3, 1), frame_out_right_middle_down.permute(0, 2, 3, 1), frame_out_down_right[:, :, 664:, 400:].permute(0, 2, 3, 1)],1))

        
        
      frame_out=torch.cat([frame_out_left, frame_out_leftmiddle, frame_out_middle, frame_out_middleright, frame_out_right],2).permute(0, 3, 1, 2)
      
      frame_out=frame_out.to(device)
      
      fourth_channel=torch.ones([frame_out.shape[0],1,frame_out.shape[2],frame_out.shape[3]],device='cuda:0')
      frame_out_rgba=torch.cat([frame_out,fourth_channel],1)
      print(frame_out_rgba.shape)
        
      name= re.findall("\d+",str(name))
      imwrite(frame_out_rgba, output_dir + '/' + str(i) +'-'+ str(name[0])+'.png', range=(0, 1))
      
    #   a, b =to_psnr(frame_out, clean)
    #   psnr_list.extend(a)
    #   c=to_ssim_skimage(frame_out, clean)
    #   ssim_list.extend(c)
    #   rmse_list.extend(b)##
    #   with open('New Checkpoints/3-dwt-ffc-original-convnext-original-lr-new-dataset/test4-larger-than-21.6.txt', 'a') as f:
    #     f.write( str(name[0])+"_psnr:" + str(a[0]) + ";" + str(name[0])+"_ssim:"+str(c[0]) +";" +str(name[0])+"_rmse:"+ str(b[0]) + ";")
    # avr_psnr = sum(psnr_list) / len(psnr_list)
    # avr_ssim = sum(ssim_list) / len(ssim_list)
    # avr_rmse = sum(rmse_list) / len(rmse_list)
    # with open('New Checkpoints/3-dwt-ffc-original-convnext-original-lr-new-dataset/test4-larger-than-21.6.txt', 'a') as f:
    #   f.write( "avr_psnr:" + str(avr_psnr) + "; avr_ssim:"+ str(avr_ssim) +"; avr_rmse:"+ str(avr_rmse) + "\n")

    # print(avr_psnr)
    # print(avr_ssim)
    
    # writer.add_scalars('testing', {'testing psnr': avr_psnr, 'testing rmse': avr_rmse,
    #                                'testing ssim': avr_ssim
    #                                }, number)

    test_time = time.time() - start_time
    print(test_time)











