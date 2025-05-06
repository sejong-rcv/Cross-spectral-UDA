import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import gc

def loss_calc(pred, label, day_mask = None, gpu=None, ignore_coarse_label=None):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    if gpu is None:
        label = Variable(label.long()).cuda()
        criterion = CrossEntropy2d(ignore_coarse_label=ignore_coarse_label, day_mask=day_mask).cuda()
    else:
        label = Variable(label.long()).cuda(gpu)
        criterion = CrossEntropy2d(ignore_coarse_label=ignore_coarse_label, day_mask=day_mask).cuda(gpu)

    return criterion(pred, label)

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, ignore_coarse_label = None, day_mask=None):
        super(CrossEntropy2d,self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.ignore_coarse_label = ignore_coarse_label
        self.day_mask = day_mask

    def forward(self, predict, target, weight=None):

        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        
        ### For stage 2
        if self.day_mask is not None:
            predict = predict[self.day_mask==1]
            target = target[self.day_mask==1]
        
        n, c, h, w = predict.size()
        
        if self.ignore_coarse_label is None:
            target_mask = (target >= 0) * (target != self.ignore_label)
        else:
            target_mask = (target >= 0) * (target != self.ignore_label) * (target < self.ignore_coarse_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous() #BxCxHxW-> BxHxWxC
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)        
        
        loss = F.cross_entropy(predict, target, weight=weight, reduction='none') #size_average=self.size_average)

        return loss.mean(), loss.reshape(n,h,w) 

                    
def softmax_kl_loss(inputs, targets, mask=None):
    assert inputs.size() == targets.size()
    
    n, c, h, w = inputs.size()
    
    inputs = inputs.transpose(1, 2).transpose(2, 3).reshape(-1,c).contiguous() #B,C,H,W-> BxHxW, C
    targets = targets.transpose(1, 2).transpose(2, 3).reshape(-1,c).contiguous() #B,C,H,W-> BxHxW, C
    
    if mask is not None:
        
        if mask.shape[2] != h:
            mask = F.interpolate(mask.float(), size=(h, w))
            mask = mask[:,0].unsqueeze(1).expand(-1,c,-1,-1)
    
        mask = mask.transpose(1, 2).transpose(2, 3).reshape(-1,c).contiguous() #B,C,H,W-> BxHxW, C
        
        input_log_softmax = F.log_softmax(inputs*mask, dim=1)
        targets = F.softmax((targets*mask).detach(), dim=1)        
        
    else:
        input_log_softmax = F.log_softmax(inputs, dim=1)
        targets = F.softmax(targets.detach(), dim=1)
    
    return F.kl_div(input_log_softmax, targets, reduction='batchmean') 


def commmon_prototype_sim_loss(total_prototype, inputs, labels, cross_entropy_weight):
    
    B, n_classes, _ = cross_entropy_weight.size()
    _, ch, inputs_h, inputs_w = inputs.size()
        
    total_prototype = total_prototype.expand(B,n_classes,ch)
    
    cs_map = torch.matmul(F.normalize(total_prototype, dim = 2), F.normalize(inputs.squeeze().resize(B,ch,inputs_h*inputs_w), dim=1))        
    cs_map[cs_map==0] = -1
    
    cosine_similarity_map = F.interpolate(cs_map.resize(B,n_classes, inputs_h, inputs_w), size=labels.size()[-2:])
    cosine_similarity_map *= 10

    loss_list = torch.zeros(size=(B,1)).cuda()
    
    for i in range(B):
        prototype_loss = torch.nn.CrossEntropyLoss(weight= cross_entropy_weight[i], ignore_index=255)
        loss = prototype_loss(cosine_similarity_map[i].unsqueeze(0), labels[i].unsqueeze(0).cuda())
        loss_list[i] = loss
    
    return loss_list.mean()


def make_common_prototype_confidence(thres1, thres2, ms_pred1, ms_pred2, pred1, pred2, labels):
    
    conf_thres_func1 =torch.nn.Threshold(threshold=thres1, value=0,inplace=True)
    conf_thres_func2 =torch.nn.Threshold(threshold=thres2, value=0,inplace=True)

    B, ch, inputs_h, inputs_w = ms_pred1.size()
    _, label_h, label_w = labels.size()
    n_classes=19
    labels_list = [torch.unique(labels[i].int()) for i in range(B)] 
    
    cross_entropy_weight = torch.zeros(size=(B,n_classes,1))
    for i in range(B):
        cross_entropy_weight[i][labels_list[i].tolist()]=1
    
    cross_entropy_weight = cross_entropy_weight.cuda()
    
    fg_mask = torch.zeros(size=(B,n_classes,label_h,label_w)).cuda()
    for i in range(B):
        for label in labels_list[i]:
            fg_mask[i][label] = (labels[i]==label)*1
    fg_mask_resize = F.interpolate(fg_mask.float(), size=(inputs_h, inputs_w))
    
    conf_rgb = torch.max(F.softmax(pred1, dim=1), dim=1)[0].unsqueeze(1)
    conf_th = torch.max(F.softmax(pred2, dim=1), dim=1)[0].unsqueeze(1)

    conf_rgb = conf_thres_func1(conf_rgb)
    conf_rgb[conf_rgb!=0] = 1.

    conf_th = conf_thres_func2(conf_th)
    conf_th[conf_th!=0] = 1.
        
    conf_rgb = F.interpolate(conf_rgb.float(), size=(inputs_h, inputs_w))
    conf_th = F.interpolate(conf_th.float(), size=(inputs_h, inputs_w))

    prototype_rgb = torch.zeros(size=(n_classes, B, ch)).cuda()
    prototype_th = torch.zeros(size=(n_classes, B, ch)).cuda()
    prototype_RT = torch.zeros(size=(n_classes, B, ch)).cuda()
    
    for cls_ in range(n_classes):
        
        prototype1 = ((fg_mask_resize[:,cls_].unsqueeze(1))*ms_pred1*conf_rgb).resize(B,ch,inputs_h*inputs_w).sum(-1)/((fg_mask_resize[:,cls_]*conf_rgb[:,0]).sum()+(1e-6))
        prototype2 = ((fg_mask_resize[:,cls_].unsqueeze(1))*ms_pred2*conf_th).resize(B,ch,inputs_h*inputs_w).sum(-1)/((fg_mask_resize[:,cls_]*conf_th[:,0]).sum()+(1e-6))
        
        w_proto1 = (fg_mask_resize[:,cls_]).sum() + (1e-6)
        w_proto2 = (fg_mask_resize[:,cls_]).sum() + (1e-6)
        
        w_ = w_proto1 + w_proto2

        prototype_rgb[cls_] = prototype1
        prototype_th[cls_] = prototype2
        prototype_RT[cls_] = prototype1*(w_proto1/w_) + prototype2*(w_proto2/w_)
    
    
    prototype_rgb = prototype_rgb.permute(1,0,2).contiguous()
    prototype_th = prototype_th.permute(1,0,2).contiguous()
    prototype_RT = prototype_RT.permute(1,0,2).contiguous()
    
    mask = cross_entropy_weight.sum(dim=0)
    prototype_rgb = prototype_rgb.sum(dim=0)/(mask+(1e-6))
    prototype_th = prototype_th.sum(dim=0)/(mask+(1e-6))
    prototype_RT = prototype_RT.sum(dim=0)/(mask+(1e-6))
    
    
    return prototype_rgb, prototype_th, prototype_RT, cross_entropy_weight


def loss_calc_night(dec_pred2, total_prototype, ms_pred2):
    
    
    n_classes, _ = total_prototype.size()
    B, ch, inputs_h, inputs_w = ms_pred2.size()

    labels_list = [torch.unique(torch.argmax(dec_pred2[i], dim=0).int()) for i in range(B)]
    
    cross_entropy_weight = torch.zeros(size=(B,n_classes,1))
    for i in range(B):
        cross_entropy_weight[i][labels_list[i].tolist()]=1
    
    cross_entropy_weight = cross_entropy_weight.cuda()
   
    total_prototype = total_prototype.expand(B,n_classes,ch)
        
    cs_map = torch.matmul(F.normalize(total_prototype, dim = 2), F.normalize(ms_pred2.squeeze().resize(B,ch,inputs_h*inputs_w), dim=1))  
    cs_map[cs_map==0] = -1
    
    cosine_similarity_map = F.interpolate(cs_map.resize(B,n_classes, inputs_h, inputs_w), size=dec_pred2.size()[-2:], mode='bilinear', align_corners=False)

    dec_pred2 = dec_pred2.detach()
    
    loss_list = torch.zeros(size=(B,1)).cuda()

    cosine_similarity_map = F.softmax(cosine_similarity_map, dim=1)
    dec_pred2 = F.softmax(dec_pred2, dim=1) 
    
    return softmax_kl_loss(cosine_similarity_map*(cross_entropy_weight.unsqueeze(-1)), dec_pred2*(cross_entropy_weight.unsqueeze(-1)))

