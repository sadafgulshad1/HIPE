
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.nn import CrossEntropyLoss
from PIL import Image, ImageDraw
from utils import Logger, worker_init_fn, get_lr, label_set_
import re
from dataset import get_training_data, get_validation_data, get_inference_data
import os, glob
import copy
from pathlib import Path
from helpers import makedir, find_high_activation_crop
import model
import push
import shutil
import random
import model_protopnet
from torch import nn
import json
import imageio
import torch.backends.cudnn as cudnn
from training import train_epoch
from validation import val_epoch
from log import create_logger
from model import (generate_model, make_data_parallel,
                   get_fine_tuning_parameters)
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
import argparse
from dataset_hyperbolic import get_son2parent, get_emb, get_dataloader
from metric import Metric
from pmath import pair_wise_cos, pair_wise_eud, pair_wise_hyp
from opts import parse_opts
n_classes = 101 ## For UCF
check_test_accu = True
perform_analysis = True



def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]

    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)

    return opt

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
# train set
def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_val_utils(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor()
    ]
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))
    temporal_transform = TemporalCompose(temporal_transform)
    label_set = label_set_()
    emb = get_emb('hyp', opt.emb_file, label_set)
    val_data, collate_fn = get_validation_data(opt.video_path,
                                               opt.annotation_path, opt.dataset,
                                               opt.input_type, opt.file_type,emb,
                                               spatial_transform,
                                               temporal_transform)
    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=False)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size//
                                                         opt.n_val_samples),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'test_original_conf.log',
                            ['epoch', 'loss', 'acc'])
    else:
        val_logger = None
    
    return (val_loader,val_logger, label_set)
def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model
def main_worker(index, opt):
    prototype_shape=(630, 128, 1, 1,1)
    load_model_name = str(opt.resume_path).split('/')[-1]
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number = int(epoch_number_str)
    
    # random.seed(opt.manual_seed)
    # np.random.seed(opt.manual_seed)
    # torch.manual_seed(opt.manual_seed)
    
    label_set = label_set_()
    emb = get_emb('hyp', opt.emb_file, label_set)
    son2parent, son, parent, parent_id , grand_parent, grand_parent_id= get_son2parent(opt.tree_file)
    
    metric = Metric(label_set, son2parent)
    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    
    base_architecture = generate_model(opt)
    model = model_protopnet.construct_Net(opt, emb, metric, base_architecture=base_architecture,
                              pretrained=True, img_size=opt.sample_size,
                              num_classes=opt.n_classes,)

    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # if opt.pretrain_path:
    #     model = load_pretrained_model(model, opt.pretrain_path, opt.model,
    #                                   opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    model = make_data_parallel(model, opt.distributed, opt.device)

    # if opt.pretrain_path:
    #     parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
    # else:
    #     parameters = model.parameters()

    # if opt.is_master_node:
    #     print(model)

    criterion = CrossEntropyLoss().to(opt.device)

    
    if not opt.no_val:
        test_loader, test_logger ,class_names = get_val_utils(opt)

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None
    
    
    save_analysis_path = os.path.join(opt.result_path,"Siblings_Biking")
    makedir(save_analysis_path)

    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

    

    log('load model from ' + str(opt.resume_path))
    log('model base architecture: ' + str(opt.arch))
    
    img_size = opt.sample_size
    prototype_shape = prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]* prototype_shape[4]

    class_specific = opt.class_specific
    
    if check_test_accu:
        log('test set size: {0}'.format(len(test_loader.dataset)))
        
        val_epoch(opt, emb, metric, opt.n_epochs, test_loader, model, criterion,
                                        opt.device, test_logger, tb_writer,
                                        opt.distributed)
        
    if perform_analysis:
        ##### SANITY CHECK
        # confirm prototype class identity
        load_img_dir = "img"

    

        # confirm prototype connects most strongly to its own class
        prototype_max_connection = torch.argmax(model.module.last_layer.weight, dim=0)
        prototype_max_connection = prototype_max_connection.cpu().numpy()


        ##### HELPER FUNCTIONS FOR PLOTTING
        def save_gif(file_path, data, heatmap=None, a=.5, b=.5):
            w = data.shape[1]
            h = data.shape[2]
            if heatmap is None:
                Scaled01 = data
                Scaled01 = Scaled01-Scaled01.min()
                Scaled01 = Scaled01/Scaled01.max()
                data = Scaled01
                data = (data* 255).astype(np.uint8)
                data = [Image.fromarray(img).rotate(-90).quantize(method=Image.MEDIANCUT) for img in data]
                im1 = Image.new("RGB", (w, h))
                im1.save(file_path,save_all=True,append_images=data, duration=60, loop =0)
            else:
                data = data * a + heatmap * b
                Scaled01 = data
                Scaled01 = Scaled01-Scaled01.min()
                Scaled01 = Scaled01/Scaled01.max()
                data = Scaled01
                data = (data* 255).astype(np.uint8)
                data = [Image.fromarray(img).rotate(-90).quantize(method=Image.MEDIANCUT) for img in data]
                im1 = Image.new("RGB", (w, h))
                im1.save(file_path,save_all=True,append_images=data, duration=60, loop =0)

        def save_gif_bbox(file_path, data, heatmap=None, a=.5, b=.5):
            w = data.shape[1]
            h = data.shape[2]
                
            data = [Image.fromarray(img).rotate(-90).quantize(method=Image.MEDIANCUT) for img in data]
            im1 = Image.new("RGB", (w, h))
            im1.save(file_path,save_all=True,append_images=data, duration=60, loop =0)

        def save_prototype(fname, epoch, index):
            src = glob.glob(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index).zfill(3)+'-'+'*.gif'))
           
            dst = fname
            
            shutil.copy(src[0], dst)
            
        def save_prototype_self_activation(fname, epoch, index):
            src = glob.glob(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original_with_self_act'+str(index).zfill(3)+'-'+'*.gif'))
           
            dst = fname   
            shutil.copy(src[0], dst)

        def draw_bbox(data,high_act_patch_indices):
            Scaled01 = data
            Scaled01 = Scaled01-Scaled01.min()
            Scaled01 = Scaled01/Scaled01.max()
            data = Scaled01
            data = (data* 255).astype(np.uint8)
            data = Image.fromarray(data)
            bbox_img_ = ImageDraw.Draw(data)
            
            bbox_img_.rectangle([(high_act_patch_indices[2], high_act_patch_indices[0]), (high_act_patch_indices[3]-1, high_act_patch_indices[1]-1)],fill=None,outline="red")
            
            return np.asarray(data)
        # load the test image and forward it through the network

        with torch.no_grad():
            for batch_idx, (img_variable, test_image_label,_) in enumerate(test_loader):
                
                if batch_idx in [9,10]:
                
                    
                    images_test = img_variable.cuda()
                    labels_test = test_image_label.cuda()  
                    apred, min_distances = model(images_test)
                    
                    dist = opt.eval_dist(apred, emb, c)
                    
                    rank = dist.sort()[1]
                    
                    logits = rank[:,0]   ##only the name is logits these are rankings
                    
                    conv_output, distances = model.module.push_forward(images_test)
                    prototype_activations = model.module.distance_2_similarity(min_distances)
                    prototype_activation_patterns = model.module.distance_2_similarity(distances)
                    if model.module.prototype_activation_function == 'linear':
                        prototype_activations = prototype_activations + max_dist
                        prototype_activation_patterns = prototype_activation_patterns + max_dist
                    tables = []
                    for i in range(logits.size(0)):
                        tables.append((logits[i], labels_test[i].item()))
                    
                    
                    
                    for img_idx in range (images_test.shape[0]):
                        if batch_idx==9 and img_idx in [123,124,125]:
                            predicted_cls = tables[img_idx][0]
                            correct_cls = tables[img_idx][1]
                            log('Predicted: ' + str(predicted_cls))
                            log('Actual: ' + str(correct_cls))
                            original_img = np.transpose(images_test[img_idx].cpu().detach().numpy(), (1,3, 2, 0))
                            save_gif(os.path.join(save_analysis_path,'original_gif'+str(batch_idx)+str(img_idx)+'.gif'),
                                    original_img)


                            
            # # ##### PROTOTYPES FROM TOP-k CLASSES
                            k = 5
                            log('Prototypes from top-%d classes:' % k)
                            
                            # topk_logits, topk_classes = torch.topk(dist[img_idx], k=k)
                            topk_classes = rank[img_idx,:k]
                            
                            # topk_classes = [model.module.child_parent_id[topk_classes.item()]+n_classes,model.module.child_grand_parent_id[topk_classes.item()]+n_classes]
                           
                            for i,c in enumerate(topk_classes):
                                makedir(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1)))

                                log('top %d predicted class: %d' % (i+1, c))
                                # log('logit of the class: %f' % topk_logits[i])
                                class_prototype_indices = np.nonzero(model.module.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
                                class_prototype_activations = prototype_activations[img_idx][class_prototype_indices]
                                _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

                                prototype_cnt = 1
                                for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
                                    prototype_index = class_prototype_indices[j]
                                    save_prototype(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1),
                                                                'top-%d_activated_prototype.gif' % prototype_cnt),
                                                start_epoch_number, prototype_index)
                                    
                                    save_prototype_self_activation(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1),
                                                                                'top-%d_activated_prototype_self_act.gif' % prototype_cnt),
                                                                start_epoch_number, prototype_index)
                                    log('prototype index: {0}'.format(prototype_index))
                                    log('activation value (similarity score): {0}'.format(prototype_activations[img_idx][prototype_index]))
                                    log('last layer connection: {0}'.format(model.module.last_layer.weight[c][prototype_index]))

                                    
                                    activation_pattern = prototype_activation_patterns[img_idx][prototype_index].detach().cpu().numpy()
                                    
                                    frames = original_img.shape[0]
                                    activation_pattern = torch.tensor(activation_pattern)
                                    activation_pattern = activation_pattern.repeat(frames,1,1)
                                    activation_pattern = activation_pattern.cpu().detach().numpy()
                                    img_stack_sm = np.zeros((frames, img_size, img_size))
                                    for idx_ in range((frames)):
                                        img = activation_pattern[idx_, :, :]
                                        img_sm = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                                        img_stack_sm[idx_, :, :] = img_sm
                                    upsampled_activation_pattern = img_stack_sm
                                    

                                    bbox_img = []
                                    overlayed_img = []
                                    act_img = []
                                    for k in range (upsampled_activation_pattern.shape[0]):
                                        
                                        # show the most highly activated patch of the image by this prototype
                                        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern[k,:,:])


                                        high_act_patch = original_img[k,high_act_patch_indices[0]:high_act_patch_indices[1],
                                                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
                                        act_img.append(high_act_patch)
                                    
                                    
                                        
                                        bbox_img_ = draw_bbox(original_img[k],high_act_patch_indices)
                                        bbox_img.append(bbox_img_)
                                        
                                        

                                        # show the image overlayed with prototype activation map
                                        rescaled_activation_pattern = upsampled_activation_pattern[k] - np.amin(upsampled_activation_pattern[k])
                                        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
                                        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
                                        heatmap = np.float32(heatmap) / 255
                                        heatmap = heatmap[...,::-1]
                                        over_img=(0.5 * original_img[k] + 1 * heatmap)
                                        overlayed_img.append(over_img)
                                    log('most highly activated patch of the chosen image by this prototype:')
                                    
                                    save_gif(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1),
                                                        'most_highly_activated_patch_by_top-%d_prototype.gif' % prototype_cnt),
                                            np.asarray(act_img))
                                    log('most highly activated patch by this prototype shown in the original image:')
                                    save_gif(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1),
                                                                        'most_highly_activated_patch_in_original_img_by_top-%d_prototype.gif' % prototype_cnt),np.asarray(bbox_img))
                                    log('prototype activation map of the chosen image:')
                                    
                                    save_gif(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1),
                                                            'prototype_activation_map_by_top-%d_prototype.gif' % prototype_cnt),
                                            np.asarray(overlayed_img))
                                    log('--------------------------------------------------------------')
                                    prototype_cnt += 1
                                log('***************************************************************')
                            
                            if predicted_cls == correct_cls:
                                log('Prediction is correct.')
                            else:
                                log('Prediction is wrong.')
                        elif batch_idx == 10 and img_idx in [0,1,2]:
                                predicted_cls = tables[img_idx][0]
                                correct_cls = tables[img_idx][1]
                                log('Predicted: ' + str(predicted_cls))
                                log('Actual: ' + str(correct_cls))
                                original_img = np.transpose(images_test[img_idx].cpu().detach().numpy(), (1,3, 2, 0))
                                save_gif(os.path.join(save_analysis_path,'original_gif'+str(batch_idx)+str(img_idx)+'.gif'),
                                        original_img)



                                # # ##### PROTOTYPES FROM TOP-k CLASSES
                                k = 5
                                log('Prototypes from top-%d classes:' % k)

                                # topk_logits, topk_classes = torch.topk(dist[img_idx], k=k)
                                topk_classes = rank[img_idx,:k]

                                # topk_classes = [model.module.child_parent_id[topk_classes.item()]+n_classes,model.module.child_grand_parent_id[topk_classes.item()]+n_classes]

                                for i,c in enumerate(topk_classes):
                                    makedir(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1)))

                                    log('top %d predicted class: %d' % (i+1, c))
                                    # log('logit of the class: %f' % topk_logits[i])
                                    class_prototype_indices = np.nonzero(model.module.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
                                    class_prototype_activations = prototype_activations[img_idx][class_prototype_indices]
                                    _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

                                    prototype_cnt = 1
                                    for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
                                        prototype_index = class_prototype_indices[j]
                                        save_prototype(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1),
                                                                    'top-%d_activated_prototype.gif' % prototype_cnt),
                                                    start_epoch_number, prototype_index)
                                        
                                        save_prototype_self_activation(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1),
                                                                                    'top-%d_activated_prototype_self_act.gif' % prototype_cnt),
                                                                    start_epoch_number, prototype_index)
                                        log('prototype index: {0}'.format(prototype_index))
                                        log('activation value (similarity score): {0}'.format(prototype_activations[img_idx][prototype_index]))
                                        log('last layer connection: {0}'.format(model.module.last_layer.weight[c][prototype_index]))

                                        
                                        activation_pattern = prototype_activation_patterns[img_idx][prototype_index].detach().cpu().numpy()
                                        
                                        frames = original_img.shape[0]
                                        activation_pattern = torch.tensor(activation_pattern)
                                        activation_pattern = activation_pattern.repeat(frames,1,1)
                                        activation_pattern = activation_pattern.cpu().detach().numpy()
                                        img_stack_sm = np.zeros((frames, img_size, img_size))
                                        for idx_ in range((frames)):
                                            img = activation_pattern[idx_, :, :]
                                            img_sm = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                                            img_stack_sm[idx_, :, :] = img_sm
                                        upsampled_activation_pattern = img_stack_sm
                                        

                                        bbox_img = []
                                        overlayed_img = []
                                        act_img = []
                                        for k in range (upsampled_activation_pattern.shape[0]):
                                            
                                            # show the most highly activated patch of the image by this prototype
                                            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern[k,:,:])


                                            high_act_patch = original_img[k,high_act_patch_indices[0]:high_act_patch_indices[1],
                                                                        high_act_patch_indices[2]:high_act_patch_indices[3], :]
                                            act_img.append(high_act_patch)
                                        
                                        
                                            
                                            bbox_img_ = draw_bbox(original_img[k],high_act_patch_indices)
                                            bbox_img.append(bbox_img_)
                                            
                                            

                                            # show the image overlayed with prototype activation map
                                            rescaled_activation_pattern = upsampled_activation_pattern[k] - np.amin(upsampled_activation_pattern[k])
                                            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
                                            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
                                            heatmap = np.float32(heatmap) / 255
                                            heatmap = heatmap[...,::-1]
                                            over_img=(0.5 * original_img[k] + 1 * heatmap)
                                            overlayed_img.append(over_img)
                                        log('most highly activated patch of the chosen image by this prototype:')
                                        
                                        save_gif(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1),
                                                            'most_highly_activated_patch_by_top-%d_prototype.gif' % prototype_cnt),
                                                np.asarray(act_img))
                                        log('most highly activated patch by this prototype shown in the original image:')
                                        save_gif(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1),
                                                                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.gif' % prototype_cnt),np.asarray(bbox_img))
                                        log('prototype activation map of the chosen image:')
                                        #plt.axis('off')
                                        save_gif(os.path.join(save_analysis_path, str(batch_idx)+str(img_idx)+'top-%d_class_prototypes' % (i+1),
                                                                'prototype_activation_map_by_top-%d_prototype.gif' % prototype_cnt),
                                                np.asarray(overlayed_img))
                                        log('--------------------------------------------------------------')
                                        prototype_cnt += 1
                                    log('***************************************************************')

                                if predicted_cls == correct_cls:
                                    log('Prediction is correct.')
                                else:
                                    log('Prediction is wrong.')
                            
    logclose()


if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()
    if opt.distributed:
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    else:
        main_worker(-1, opt)