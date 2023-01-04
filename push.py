import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
from PIL import Image
import imageio
from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop
hierarchical_push = False
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
        im1 = Image.new("RGB", (w,h))
        im1.save(file_path,save_all=True,append_images=data, duration=60, loop =0)
# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3],prototype_shape[4]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, search_y,_) in enumerate(dataloader):

        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    ### UPDATING PROTOTYPE IN THE NETWORK ###
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))
def push_operation(j,proto_dist_j,global_min_proto_dist,class_specific,class_to_img_index_dict,target_class,
                   prototype_layer_stride,proto_h,proto_w,proto_t,protoL_input_,global_min_fmap_patches,
                   search_batch_input,proto_dist_,prototype_network_parallel,max_dist,
                   prototype_activation_function_in_numpy,dir_for_saving_prototypes,
                   prototype_img_filename_prefix,prototype_self_act_filename_prefix):
    batch_min_proto_dist_j = np.amin(proto_dist_j)
    if batch_min_proto_dist_j < global_min_proto_dist[j]:
        batch_argmin_proto_dist_j = \
            list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                    proto_dist_j.shape))
        if class_specific:
            '''
            change the argmin index from the index among
            images of the target class to the index in the entire search
            batch
            '''
            batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

        # retrieve the corresponding feature map patch
        img_index_in_batch = batch_argmin_proto_dist_j[0]
        fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
        fmap_height_end_index = fmap_height_start_index + proto_h
        fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
        fmap_width_end_index = fmap_width_start_index + proto_w
        fmap_time_start_index = batch_argmin_proto_dist_j[3] * prototype_layer_stride
        fmap_time_end_index = fmap_time_start_index + proto_t

        batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,:,
                                                fmap_height_start_index:fmap_height_end_index,
                                                fmap_width_start_index:fmap_width_end_index,fmap_time_start_index:fmap_time_end_index]

        global_min_proto_dist[j] = batch_min_proto_dist_j
        global_min_fmap_patches[j] = batch_min_fmap_patch_j

        

        # get the whole image
        original_img_j = search_batch_input[img_index_in_batch].cpu().clone()

        AA = original_img_j.view(original_img_j.size(0), -1)
        AA -= AA.min(1, keepdim=True)[0]
        AA /= AA.max(1, keepdim=True)[0]
        original_img_j = AA.view(3, 16, 112, 112)
        original_img_j = original_img_j.numpy()
        original_img_j = np.transpose(original_img_j, (1,3, 2, 0))

        original_img_size = original_img_j.shape[1]



        proto_dist_img_j = proto_dist_[img_index_in_batch, j,:, :, :]

        if prototype_network_parallel.module.prototype_activation_function == 'log':
            proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.module.epsilon))
        elif prototype_network_parallel.module.prototype_activation_function == 'linear':
            proto_act_img_j = max_dist - proto_dist_img_j
        else:
            proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)


        
        frames = original_img_j.shape[0]
        proto_act_img_j = torch.tensor(proto_act_img_j)
        proto_act_img_j = proto_act_img_j.repeat(frames,1,1)
        # print (proto_act_img_j.shape)
        proto_act_img_j = proto_act_img_j.cpu().detach().numpy()
        img_stack_sm = np.zeros((frames, original_img_size, original_img_size))
        for idx in range((frames)):
            img = proto_act_img_j[idx, :, :]
            img_sm = cv2.resize(img, (original_img_size, original_img_size), interpolation=cv2.INTER_CUBIC)
            img_stack_sm[idx, :, :] = img_sm
        upsampled_act_img_j = img_stack_sm

        proto_img_j_gif = []
        overlayed_original_img_j_gif = []
        for k in range (upsampled_act_img_j.shape[0]):
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j[k,:,:])

            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[k,proto_bound_j[0]:proto_bound_j[1],
                                            proto_bound_j[2]:proto_bound_j[3], :]



            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                            prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    

                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j[k] - np.amin(upsampled_act_img_j[k])
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)

                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255

                    heatmap = heatmap[...,::-1]


                    overlayed_original_img_j = 0.5 * original_img_j[k] + 0.3 * heatmap
                    

                    overlayed_original_img_j_gif.append(overlayed_original_img_j)
                    
                    proto_img_j_gif.append(proto_img_j)
        save_gif(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-original'+ str(j).zfill(3)+'-'+str(target_class).zfill(3)+'.gif'),
                    original_img_j)
        save_gif(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-original_with_self_act' + str(j).zfill(3)+'-'+str(target_class).zfill(3)+ '.gif'),
                        np.asarray(overlayed_original_img_j_gif))
        
        save_gif(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + str(j).zfill(3) +'-'+str(target_class).zfill(3)+'.gif'),
                    np.asarray(proto_img_j_gif))
# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
    
    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes+prototype_network_parallel.module.num_parents+prototype_network_parallel.module.num_grand_parents)}        
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index) 
            ## Image to Id for parent classes
            if img_label in prototype_network_parallel.module.child_parent_id.keys():               
                class_to_img_index_dict[prototype_network_parallel.module.child_parent_id[img_label]+num_classes].append(img_index)
            ## Image to Id for grand parent classes
            if img_label in prototype_network_parallel.module.child_grand_parent_id.keys():                    
                class_to_img_index_dict[prototype_network_parallel.module.child_grand_parent_id[img_label]+num_classes].append(img_index)
        
        
        
            
        
        
        
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    proto_t = prototype_shape[4]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]* prototype_shape[4]
    proto_per_class = n_prototypes//num_classes
    
    
    for j in range(n_prototypes):
        #if n_prototypes_per_class != None:
        
        
        if class_specific:
            
            
            if hierarchical_push: 
                target_classes = ((prototype_network_parallel.module.prototype_parent_child_identity[j]==1).nonzero().squeeze())
                
                target_class, target_parent_class, target_grand_parent_class = target_classes[0], target_classes[1], target_classes[2]
                
                if len(class_to_img_index_dict[target_class.item()]) == 0:
                    continue
                
                proto_dist_j = proto_dist_[class_to_img_index_dict[target_class.item()]][:,j,:,:,:]
                push_operation(j,proto_dist_j,global_min_proto_dist,class_specific,class_to_img_index_dict,target_class.item(),
                prototype_layer_stride,proto_h,proto_w,proto_t,protoL_input_,global_min_fmap_patches,
                search_batch_input,proto_dist_,prototype_network_parallel,max_dist,
                prototype_activation_function_in_numpy,dir_for_saving_prototypes,
                prototype_img_filename_prefix,prototype_self_act_filename_prefix)
                        
            else:
                # target_class is the class of the class_specific prototype
                target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
                # if there is not images of the target_class from this batch
                # we go on to the next prototype
                if len(class_to_img_index_dict[target_class]) == 0:
                    continue
                proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:,:]
                push_operation(j,proto_dist_j,global_min_proto_dist,class_specific,class_to_img_index_dict,target_class,
                   prototype_layer_stride,proto_h,proto_w,proto_t,protoL_input_,global_min_fmap_patches,
                   search_batch_input,proto_dist_,prototype_network_parallel,max_dist,
                   prototype_activation_function_in_numpy,dir_for_saving_prototypes,
                   prototype_img_filename_prefix,prototype_self_act_filename_prefix)
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:,j,:,:,:]
            push_operation(j,proto_dist_j,global_min_proto_dist,class_specific,class_to_img_index_dict,target_class,
                   prototype_layer_stride,proto_h,proto_w,proto_t,protoL_input_,global_min_fmap_patches,
                   search_batch_input,proto_dist_,prototype_network_parallel,max_dist,
                   prototype_activation_function_in_numpy,dir_for_saving_prototypes,
                   prototype_img_filename_prefix,prototype_self_act_filename_prefix)
    
        
    if class_specific:
        del class_to_img_index_dict
