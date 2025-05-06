from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import transforms
import gc
import tqdm 
import argparse
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime

import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from options.print import print_options
from model.SegNet import SegNet
from utils.augmentation import RandomFlip_KP_fake, RandomFlip_MF_fake
from loss.loss import *
from utils.utils import set_requires_grad, adjust_learning_rate, \
    load_state_from_model, visualize, eval_seg, save_file
    
from options.parser_ import _Parser

parser = argparse.ArgumentParser(description='default for MSG models')

parser.add_argument('--is_continue', default=False, action="store_true", help='restart the training')
parser.add_argument('--start_epo', default=0)
parser.add_argument('--config', '-c', default='configs/MFD_fake.yaml', help='path to the config file')
parser.add_argument('--test_mode', action="store_true", default=False)
parser.add_argument('--TBP_cycle', default=10) # tensorboard cycle
parser.add_argument('--save_cycle', default=20) # ckpt save cycle
parser.add_argument('--kl_weight', default=1) # loss weight of masked mutual learning
parser.add_argument('--ExpGamma', default=0.99) # gamma value of lr scheduler (ExponentialLR)

args = parser.parse_args()

Parser = _Parser()
args = Parser.gather_option(args, parser)
print_options(args, parser)

if args.dataset =='KPdataset':
    augmentation_methods = [RandomFlip_KP_fake(prob=0.5)]
    from datasets.KPD_train import KP_dataset as dataset
    from datasets.KPD_test import KP_dataset_test
        
elif args.dataset =='MFdataset':
    augmentation_methods = [RandomFlip_MF_fake(prob=0.5)]
    from datasets.MFD_train import MF_dataset as dataset
    from datasets.MFD_test import MF_dataset_test

if __name__ == '__main__':

    cudnn.enabled = True

    if args.dataset == 'KPdataset':
        H, W = 512, 640
        trainloader = DataLoader(
            dataset(args.root_dir + args.data_dir, input_folder='pseudo_KP',transform=augmentation_methods),
                batch_size=args.train_batch,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True
        )
        testloader = DataLoader(
            KP_dataset_test(args.root_dir + args.data_dir),
                batch_size=args.test_batch,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False
        )
    else:
        H, W = 480, 640
        trainloader = DataLoader(
            dataset(args.root_dir + args.data_dir, transform=augmentation_methods),
                batch_size=args.train_batch,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False 
        )
        testloader = DataLoader(
            MF_dataset_test(args.root_dir + args.data_dir),
                batch_size=args.test_batch,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
        )
        
    NUM_DATASET = len(trainloader.dataset)
    NUM_TEST_DATASET = len(testloader.dataset)
    print("train data : {} images".format(NUM_DATASET))
    print("test data : {} images".format(NUM_TEST_DATASET))
    
    model_name = args.model
    
    model = SegNet(args, num_layers=args.num_layers)  # initialize model, send model to device
    
    log_dir = os.path.join(args.root_dir, args.log_dir, model_name)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(log_dir) and args.tensorboard:
        writer = SummaryWriter(log_dir)
    if os.path.exists(log_dir):
        record_filename = os.path.join(log_dir, 'loss_record_per_iter.txt')
        record_epoch_filename = os.path.join(log_dir, 'loss_recopord_per_epoch.txt')
        with open(record_filename, 'a') as newtxt:
            newtxt.write(str(datetime.today()) + '\n')
        with open(record_epoch_filename, 'a') as epochtxt:
            epochtxt.write(str(datetime.today()) + '\n')

    check_dir = os.path.join(args.root_dir, args.check_dir, model_name)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

    if args.test_mode is True:
        model_file = os.path.join(check_dir, model_name + '_best_model.pth')
    elif args.is_continue:
        model_file = os.path.join(check_dir, model_name + '_' + str(args.start_epo) + '.pth')
    pretrained_weight = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(0))
    load_state_from_model(pretrained_weight, model)
    
    G1 = model.net_G_rgb
    G2 = model.net_G_thermal
    Decoder = model.decoder

#### Test Mode
    if args.test_mode is True: 
        print('Start test')
        print_dict = eval_seg(args.dataset, testloader, model_name, NUM_TEST_DATASET, G2, Decoder, H)
        print("All_ {} images, {} model, epoch {} checkpoint : ".format(NUM_TEST_DATASET, model_name, int(args.start_epo)))
        print(print_dict)
         
#### Train Mode   
    else:
        optimizer_G1 = optim.AdamW(G1.parameters(), lr=args.base_lr_G)
        optimizer_G2 = optim.AdamW(G2.parameters(), lr=args.base_lr_G)
        optimizer_Dec = optim.AdamW(Decoder.parameters(), lr=args.base_lr_Dec)
        
        if args.is_continue:
            G1_optim = os.path.join(check_dir, model_name + '_' + str(args.start_epo) + '_G1.optim')
            G2_optim = os.path.join(check_dir, model_name + '_' + str(args.start_epo) + '_G2.optim')
            Dec_optim = os.path.join(check_dir, model_name + '_' + str(args.start_epo) + '_Dec.optim')
            load_G1 = torch.load(G1_optim, map_location=lambda storage, loc: storage.cuda(0))
            optimizer_G1.load_state_dict(load_G1)
            del load_G1
            load_G2 = torch.load(G2_optim, map_location=lambda storage, loc: storage.cuda(0))
            optimizer_G2.load_state_dict(load_G2)
            del load_G2
            load_Dec = torch.load(Dec_optim, map_location=lambda storage, loc: storage.cuda(0))
            optimizer_Dec.load_state_dict(load_Dec)
            del load_Dec
            
        if args.is_continue:
            start_epoch = int(args.start_epo) 
            total_iters = start_epoch * NUM_DATASET
            print_dict = eval_seg(args.dataset, testloader, model_name, NUM_TEST_DATASET, G2, Decoder, H, train_mode=True)
            best_mIOU = print_dict['mean IoU']
            
        else:
            start_epoch = 0
            total_iters = 0
            best_mIOU = 0
            print_dict={}
            print_dict['mean IoU']=0 
            
        max_iters = NUM_DATASET * args.max_epoch
        print("max_iteration = {}".format(max_iters))

        kl_loss = nn.KLDivLoss(reduction='batchmean') 
        
        scheduler_G1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_G1, gamma=float(args.ExpGamma), last_epoch=-1)
        scheduler_G2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_G2, gamma=float(args.ExpGamma), last_epoch=-1)
        scheduler_Dec = torch.optim.lr_scheduler.ExponentialLR(optimizer_Dec, gamma=float(args.ExpGamma), last_epoch=-1)

        for epoch in tqdm.tqdm(range(start_epoch, args.max_epoch)):
            
            # set training model
            G1.train()
            G2.train()
            Decoder.train()
            set_requires_grad([G1, G2, Decoder], True)
            
            kl_weight = float(args.kl_weight)
        
            epoch_start_time = time.time()
            
            for i, batch in enumerate(trainloader):
                iter_start_time = time.time()

                optimizer_G1.zero_grad()
                optimizer_G2.zero_grad()
                optimizer_Dec.zero_grad()
                    
                if args.dataset == 'KPdataset':
                    rgb_images, th_images, labels, names, fake = batch #KP
                else :
                    rgb_images, th_images, gts, labels, names, fake = batch #MF

                if np.random.rand() < 0.5: # fake thermal night probability = 0.5
                    del th_images
                    th_images = Variable(fake).cuda()
                    for n_idx in range(len(names)):
                        names[n_idx] = names[n_idx] + '_2N'
                else:
                    del fake
                    th_images = Variable(th_images).cuda()
                rgb_images = Variable(rgb_images).cuda()
                
            # 1. Encoder/Decoder 
                mid_pred1 = G1(rgb_images)
                ms_pred1, pred1 = Decoder(mid_pred1)
                
                mid_pred2 = G2(th_images)
                ms_pred2, pred2 = Decoder(mid_pred2)
            
                pred1_seg_loss_rgb, pixel_loss1 = loss_calc(pred1, labels)
                pred2_seg_loss_th, pixel_loss2 = loss_calc(pred2, labels)
                  
            # 2. Masked Mutual Learning          
                pixel_loss = torch.cat((pixel_loss1.unsqueeze(1), pixel_loss2.unsqueeze(1)), dim=1).detach() 
                pixel_loss = pixel_loss.permute(0,2,3,1).contiguous().reshape(rgb_images.size(0),-1,2) 
                mask_ = F.softmax((-1)*(pixel_loss),dim=2).permute(0,2,1).contiguous().reshape(rgb_images.size(0),2,H,W) # inter spectrum 
                
                sig_pixel_loss = (1-F.sigmoid(pixel_loss))*2 # intra spectrum
                mask = mask_*(sig_pixel_loss.permute(0,2,1).contiguous().reshape(rgb_images.size(0),2,H,W))
                
                rgb_mask, th_mask = mask[:,0], mask[:,1] 
                    
                th_mask = th_mask.unsqueeze(1).expand_as(pred1) 
                pred1_kl_loss_rgb = softmax_kl_loss(pred1, pred2, th_mask)
                
                rgb_mask = rgb_mask.unsqueeze(1).expand_as(pred2)
                pred2_kl_loss_th = softmax_kl_loss(pred2, pred1, rgb_mask)
                
            # 3. MS Prototypes 
                _, _, prototype_RT, cross_entropy_weight = \
                                make_common_prototype_confidence(0.1, 0.1, ms_pred1, ms_pred2, pred1.clone().detach(), pred2.clone().detach(), labels)
                            
                #### EMA 
                if total_iters == 0:
                    total_prototype = (prototype_RT.clone()).detach()
                else:
                    total_prototype = ((total_prototype.detach())*0.9 + prototype_RT*0.1).detach()   

                proto_seg_loss_rgb = commmon_prototype_sim_loss(total_prototype, ms_pred1, labels, cross_entropy_weight)
                proto_seg_loss_th = commmon_prototype_sim_loss(total_prototype, ms_pred2, labels, cross_entropy_weight)             
                
            # Total Loss
                loss_rgb = pred1_seg_loss_rgb + proto_seg_loss_rgb*0.2 + pred1_kl_loss_rgb*kl_weight
                loss_th = pred2_seg_loss_th + proto_seg_loss_th*0.2 + pred2_kl_loss_th*kl_weight
                
                total_loss = loss_rgb + loss_th
                total_loss.backward()
                
                optimizer_G1.step()
                optimizer_G2.step()
                optimizer_Dec.step()
                                   
                scalar_info_rgb = {
                    'G1/Seg_loss': pred1_seg_loss_rgb.item(),
                    'G1/KL_loss': pred1_kl_loss_rgb.item(),
                }
                scalar_info_th = {
                    'G2/Seg_loss': pred2_seg_loss_th.item(),
                    'G2/KL_loss': pred2_kl_loss_th.item(),
                }   
                scalar_info_proto = {
                    'Proto/Rgb_proto_loss': proto_seg_loss_rgb.item(),
                    'Proto/Th_proto_loss': proto_seg_loss_th.item(),
                }
                scalar_info_lr = {
                    'lr/G1_G2': optimizer_G1.param_groups[0]['lr'],
                    'lr/Dec' : optimizer_Dec.param_groups[0]['lr'],
                }              
            
                if args.tensorboard and i%len(trainloader)//2 ==0:
                    for key, val in scalar_info_lr.items():
                        writer.add_scalar(key, val, total_iters)
                    for key, val in scalar_info_rgb.items():
                        writer.add_scalar(key, val, total_iters)
                    for key, val in scalar_info_th.items():
                        writer.add_scalar(key, val, total_iters)
                    for key, val in scalar_info_proto.items():
                        writer.add_scalar(key, val, total_iters)
                    
                total_iters += len(batch[0])
                
                if i % 100 == 0 and (epoch % int(args.save_cycle)==int(args.save_cycle)-1 or epoch == (args.max_epoch)-1):
                    if args.dataset == 'KPdataset': #KP
                        gts = None
                    visualize(rgb_images, th_images, gts, labels, pred1, pred2, names, check_dir, epoch=epoch+1)
                    print('visualize predictions ...')
                if total_iters == 0:
                    print('save test model ...')
                    torch.save(model.state_dict(), os.path.join(check_dir, model_name + '_' + 'test' + '.pth'))
              
            scheduler_G1.step()
            scheduler_G2.step()
            scheduler_Dec.step()
            
            if args.tensorboard:
                if (epoch % int(args.TBP_cycle)==int(args.TBP_cycle)-1 or epoch == (args.max_epoch)-1):

                    print_dict = eval_seg(args.dataset, testloader, model_name, NUM_TEST_DATASET, G2, Decoder, H, train_mode=True)
                    print("Thermal Stream, {} images, {} model, epoch {} checkpoint : ".format(NUM_TEST_DATASET, model_name, epoch+1))
                    print(print_dict)
                    test_info = {
                        'test/mIoU': print_dict['mean IoU'],
                        'test/car': print_dict['IoU car'],
                        'test/person': print_dict['IoU person'],
                        'test/bicycle': print_dict['IoU bicycle'],
                    }
                    for key, val in test_info.items():
                        writer.add_scalar(key, val, total_iters)
                    
                    if args.dataset == 'KPdataset':
                        print_dict = eval_seg(args.dataset, testloader, model_name, len(testloader.dataset), G2, Decoder, H, train_mode=True)
                        print("split test, {} images, {} model, epoch {} checkpoint : ".format(len(testloader.dataset), model_name, epoch+1))
                        print(print_dict)
                        test_info = {
                            'test/mIoU': print_dict['mean IoU'],
                            'test/car': print_dict['IoU car'],
                            'test/person': print_dict['IoU person'],
                            'test/bicycle': print_dict['IoU bicycle'],
                        }
                        for key, val in test_info.items():
                            writer.add_scalar(key, val, total_iters)
            
            batch_mIOU = print_dict['mean IoU']
                
            if best_mIOU < batch_mIOU:
                best_mIOU = batch_mIOU
                torch.save(model.state_dict(), os.path.join(check_dir, model_name + '_best_model.pth'))
                np.save(os.path.join(check_dir, 'total_prototype.npy'), total_prototype.cpu().numpy())
            

            if epoch==0:
                print("code save")
                save_file(args, os.path.abspath( __file__ ))
            
            if (epoch>args.max_epoch//2) and (epoch%int(args.save_cycle)==int(args.save_cycle)-1):
                print('save model ...')
                # torch.save(optimizer_G1.state_dict(), os.path.join(check_dir, model_name + '_' + str(epoch+1) + '_G1.optim'))
                # torch.save(optimizer_G2.state_dict(), os.path.join(check_dir, model_name + '_' + str(epoch+1) + '_G2.optim'))
                # torch.save(optimizer_Dec.state_dict(), os.path.join(check_dir, model_name + '_' + str(epoch+1) + '_Dec.optim'))
                torch.save(model.state_dict(), os.path.join(check_dir, model_name + '_latest_model.pth'))
            print('time taken: %.3f sec per epoch' % (time.time() - epoch_start_time))

        if args.tensorboard:
            writer.close()