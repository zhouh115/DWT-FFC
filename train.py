import torch
import time
import argparse
from model_convnext import fusion_net, Discriminator
from train_dataset_384 import dehaze_train_dataset
from test_dataset_for_testing import dehaze_test_dataset
from torch.utils.data import DataLoader
import os
from utils_test import to_psnr, to_ssim_skimage
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim
from perceptual import LossNetwork

from tqdm import tqdm

# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='DWT-FFC')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=2, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='Checkpoints/')
parser.add_argument('--log_dir', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--output_dir', type=str, default='Output/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
args = parser.parse_args()
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
train_dataset = os.path.join('Datasets-NTIRE2023-40/Train/')   # update your  path

# --- test --- #
test_dataset = os.path.join('Datasets-NTIRE2023-40/Test/')   # update your  path
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device(device_ids[0])
device = 'cuda'
print(device)

# --- Define the network --- #
MyEnsembleNet = fusion_net()
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))
DNet = Discriminator()

# # --- Load the network weight if you want to resume your training --- #
# try:
#     MyEnsembleNet.load_state_dict(torch.load(os.path.join('weights/', 'epoch5460.pkl')))
#     print('--- weight loaded ---')
# except:
#     print('--- no weight loaded ---')

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=1e-4)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[3000, 5000, 8000], gamma=0.5)
D_optim = torch.optim.Adam(DNet.parameters(), lr=1e-4)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[5000, 7000, 8000], gamma=0.5)


# --- Load training data --- #
dataset = dehaze_train_dataset(train_dataset)
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)

# --- Load testing data --- #
test_dataset = dehaze_test_dataset(test_dataset)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

MyEnsembleNet = MyEnsembleNet.to(device)
DNet = DNet.to(device)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True)
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model)
loss_network.eval()
msssim_loss = msssim

writer = SummaryWriter('runs')

best_psnr=0
best_ssim=0

# --- Strat training --- #
iteration = 0
for epoch in range(train_epoch+1):
    updated = False
    model_save = False
    print('we are training on epoch:',str(epoch))
    start_time = time.time()
    scheduler_G.step()
    scheduler_D.step()
    MyEnsembleNet.train()
    DNet.train()
    pbar = tqdm(train_loader)
    for batch_idx, (hazy, clean) in enumerate(pbar):
        iteration += 1
        hazy = hazy.to(device)
        clean = clean.to(device)
        output = MyEnsembleNet(hazy)
        DNet.zero_grad()
        real_out = DNet(clean).mean()
        fake_out = DNet(output).mean()
        D_loss = 1 - real_out + fake_out
        D_loss.backward(retain_graph=True)
        adversarial_loss = torch.mean(1 - fake_out)
        MyEnsembleNet.zero_grad()
        adversarial_loss = torch.mean(1 - fake_out)
        smooth_loss_l1 = F.smooth_l1_loss(output, clean)
        perceptual_loss = loss_network(output, clean)
        msssim_loss_ = -msssim_loss(output, clean, normalize=True)
        total_loss = smooth_loss_l1 + 0.01 * perceptual_loss + 0.0005 * adversarial_loss + 0.2 * msssim_loss_
        total_loss.backward()
        D_optim.step()
        G_optimizer.step()
        writer.add_scalars('training', {'training total loss': total_loss.item()
                                        }, iteration)
        writer.add_scalars('training_img', {'img loss_l1': smooth_loss_l1.item(),
                                            'perceptual': perceptual_loss.item(),
                                            'msssim': msssim_loss_.item()
                                            }, iteration)
        writer.add_scalars('GAN_training', {'d_loss': D_loss.item(),
            'd_score': real_out.item(),
            'g_score': fake_out.item()
        }, iteration)
    if epoch % 100 == 0:
        torch.save(MyEnsembleNet.state_dict(), os.path.join(args.model_save_dir, 'epoch' + str(epoch) + '.pkl'))





