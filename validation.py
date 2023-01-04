import torch
import time
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from utils import AverageMeter, calculate_accuracy
coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'clst_p': 0.5,
    'clst_gp': 0.3,
    'sep': -0.08,
    'sep_p': -0.11,
    'sep_gp': -0.13,
    'l1': 1e-4,
}
num_classes = 101
def loss_fn(support, query, dist_func, c, T):
    #Here we use synthesised support.
    logits = -dist_func(support,query,c) / T
    
    fewshot_label = torch.arange(support.size(0)).cuda()
    
    loss = F.cross_entropy(logits, fewshot_label)
    
    return loss
def eval_class_acc(model, val_loader, eval_dist, dist_func, emb, metric, c, T, flag):
    GT_list = [] # ground truth in validation order
    ypred_list = []    
        
    for i, (xbatch, ybatch, abatch) in enumerate(val_loader):
        
        xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
        apred,_ = model(xbatch)
        # ypred for recognition
        
        dist = eval_dist(apred, emb, c)
        rank = dist.sort()[1]
        ypred = rank[:,0]  
        ypred_list.append(ypred)    
        GT_list.append(ybatch)
    GT = torch.cat(GT_list)  
    ypred = torch.cat(ypred_list) 
    hop0_acc = (ypred == GT).float().mean().item() 
    hop1_acc = metric.hop_acc(ypred,GT, hops = 2)
    hop2_acc = metric.hop_acc(ypred,GT, hops = 4)
    print ("hop1 acc is",hop1_acc) 
    print ("hop2 acc is",hop2_acc)      
    return hop0_acc
def val_epoch(opt, emb, metric, epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              tb_writer=None,
              distributed=False):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_cluster_cost_p = 0
    total_cluster_cost_gp = 0
    total_avg_diversity_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_separation_cost_p = 0
    total_avg_separation_cost_p = 0
    total_separation_cost_gp = 0
    total_avg_separation_cost_gp = 0
    with torch.no_grad():
        for i, (inputs, targets, abatch) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            abatch = abatch.to(device)
            targets = targets.to(device, non_blocking=True)
            output, min_distances = model(inputs)
            
            
            l0 = loss_fn(output, abatch, opt.dist_func, opt.c, opt.T)
            if opt.class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3]* model.module.prototype_shape[4])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            # calculate cluster cost
            parent_targets = []
            grand_parent_targets = []
            for j in targets:
                j = j.item()
                if j in model.module.child_parent_id.keys():
                    
                    parent_targets.append(model.module.child_parent_id[j]+num_classes)
                if j in model.module.child_grand_parent_id.keys():   
                                    
                    grand_parent_targets.append(model.module.child_grand_parent_id[j]+num_classes)
            parent_targets = torch.LongTensor(parent_targets).cuda()
            
            
            grand_parent_targets = torch.LongTensor(grand_parent_targets).cuda()
            
            
            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,targets]).cuda()
            inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances)
            prototypes_of_correct_class_p = torch.t(model.module.prototype_class_identity[:,parent_targets]).cuda()
            
            
            inverted_distances_p, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class_p, dim=1)
            
            
            cluster_cost_parent = torch.mean(max_dist - inverted_distances_p)
            prototypes_of_correct_class_gp = torch.t(model.module.prototype_class_identity[:,grand_parent_targets]).cuda()
            inverted_distances_gp, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class_gp, dim=1)
            cluster_cost_grand_parents = torch.mean(max_dist - inverted_distances_gp)
            # calculate separation cost
            
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
            prototypes_of_wrong_class_p = 1 - prototypes_of_correct_class_p
            inverted_distances_to_nontarget_prototypes_p, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class_p, dim=1)
            separation_cost_parents = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes_p)
            prototypes_of_wrong_class_gp = 1 - prototypes_of_correct_class_gp
            inverted_distances_to_nontarget_prototypes_gp, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class_gp, dim=1)
            separation_cost_grand_parents = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes_gp)


            # calculate avg cluster cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)
            avg_separation_cost_p = \
                torch.sum(min_distances * prototypes_of_wrong_class_p, dim=1) / torch.sum(prototypes_of_wrong_class_p, dim=1)
            avg_separation_cost_p = torch.mean(avg_separation_cost_p)
            avg_separation_cost_gp = \
                torch.sum(min_distances * prototypes_of_wrong_class_gp, dim=1) / torch.sum(prototypes_of_wrong_class_gp, dim=1)
            avg_separation_cost_gp = torch.mean(avg_separation_cost_gp)

            if opt.use_l1_mask:
                
                l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
            else:

                l1 = model.module.last_layer.weight.norm(p=1)

        else:
                
            min_distance, _ = torch.min(min_distances, dim=1)
            cluster_cost = torch.mean(min_distance)
            l1 = model.module.last_layer.weight.norm(p=1)

        n_batches += 1
        total_cross_entropy += l0.item()
        total_cluster_cost += cluster_cost.item()
        total_cluster_cost_p += cluster_cost_parent.item()
        total_cluster_cost_gp += cluster_cost_grand_parents.item()
        if opt.class_specific:
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            total_separation_cost_p += separation_cost_parents.item()
            total_avg_separation_cost_p += avg_separation_cost_p.item()
            total_separation_cost_gp += separation_cost_grand_parents.item()
            total_avg_separation_cost_gp += avg_separation_cost_gp.item()

        # compute gradient and do SGD step
    
        if opt.class_specific:
            if coefs is not None:
                
                loss = (coefs['crs_ent'] * l0
                        + coefs['clst'] * cluster_cost
                        + coefs['clst_p'] * cluster_cost_parent
                        + coefs['clst_gp'] * cluster_cost_grand_parents
                        + coefs['sep'] * separation_cost
                        + coefs['sep_p'] * separation_cost_parents
                        + coefs['sep_gp'] * separation_cost_grand_parents
                        + coefs['l1'] * l1)
            else:
                
                loss = l0 + 0.8 * cluster_cost + 0.8 * cluster_cost_parent+ 0.8 * cluster_cost_grand_parents- 0.08 * separation_cost + 1e-4 * l1
        else:
            print ("Non class specific loss")
            if coefs is not None:
                loss = (coefs['crs_ent'] * l0
                        + coefs['clst'] * cluster_cost
                        + coefs['clst_p'] * cluster_cost_parent
                        + coefs['clst_gp'] * cluster_cost_grand_parents
                        + coefs['l1'] * l1)
            else:
                loss = l0 + 0.8 * cluster_cost+ 0.8 * cluster_cost_parent + 0.8 * cluster_cost_grand_parents  + 1e-4 * l1
        
            # losses.update(loss, inputs.size(0))
            
            # accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()
        if epoch % 10 == 0:
            with torch.no_grad():
                acc = eval_class_acc(model, data_loader, opt.eval_dist, opt.dist_func, emb, metric, opt.c, opt.T, flag = 'valid')
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss:.4f} ({loss:.4f})\t'
                    'Acc {acc:.3f} ({acc:.3f})'.format(
                        epoch,
                        i + 1,
                        len(data_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=loss,
                        acc=acc))


            if logger is not None:
                logger.log({'epoch': epoch, 'loss': loss, 'acc': acc})

            if tb_writer is not None:
                tb_writer.add_scalar('val/loss', loss, epoch)
                tb_writer.add_scalar('val/acc', acc, epoch)

    return loss