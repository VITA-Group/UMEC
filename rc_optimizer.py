import argparse
import os
import pickle
import sys
import time
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from torch.nn.utils import clip_grad_norm_

from rc_utils import RC_CP_MiniMax, \
                     flops2, \
                     prox_w, \
                     proj_dual

def rc_optimizer(optimizer, minimax_model, s_optimizer, dual_optimizer, args, infos,save_budgets):

    s_max = (minimax_model.s_ub - float(args.punit) - 1e-8).clamp(min=0.0)
    
    prox_w(minimax_model, optimizer)
    s_loss1 = minimax_model.sloss1(optimizer)
    s_loss2 = minimax_model.sloss2(budget=args.budget)
    cur_resource = s_loss2.item() + args.budget


    s_grad1 = torch.autograd.grad(s_loss1, minimax_model.s, only_inputs=True)[0].data \
              + args.sl2wd * (minimax_model.s.data / minimax_model.s_ub)  # >=0
    s_grad2 = torch.autograd.grad(s_loss2, minimax_model.s, only_inputs=True)[0].data  # <=0
    
    if args.zclamp:
        idx = minimax_model.s.data < s_max
        z_max = ((s_grad1.clamp(min=0.0) + 1e-8) / -s_grad2.clamp(max=0.0))[s_grad2 != 0.0].max().item()
        assert z_max >= 0
        minimax_model.z.data.clamp_(max=z_max)

    s_optimizer.zero_grad()
    minimax_model.s.grad = s_grad1 + minimax_model.z.data * s_grad2

    overflow_idx = minimax_model.s.data >= s_max
    underflow_idx = minimax_model.s.data <= 0

    minimax_model.s.grad.data[overflow_idx] = minimax_model.s.grad.data[overflow_idx].clamp(min=0.0)
    minimax_model.s.grad.data[underflow_idx] = minimax_model.s.grad.data[underflow_idx].clamp(max=0.0)
    
    clip_grad_norm_(minimax_model.s, 1.0, float('inf'))
    s_optimizer.step()
    
    minimax_model.s.data.clamp_(min=0.0)
    minimax_model.s.data[overflow_idx] = s_max[overflow_idx]

    # dual update
    dual_loss = -(minimax_model.yloss(optimizer) + minimax_model.zloss(budget=args.budget))
    dual_optimizer.zero_grad()
    dual_loss.backward()

    if args.early_stop and minimax_model.z.grad.item()>=0.0:
        return False

    dual_optimizer.step()
    proj_dual(minimax_model)

    if args.verbose:

        epoch, iter_index, loss_avg=infos['epoch'], infos['iter_index'], infos['loss_avg']

        if iter_index % args.log_interval == 0:
            s = minimax_model.ceiled_s()
            rc_cost = minimax_model.resource_fn(s.data).item()
            print('resource cost: {:.4e})'.format(rc_cost))
            print('  s_ub={}'.format(array1d_repr(minimax_model.s_ub, format='{:4.3f}')))
            print(' s_val={}'.format(array1d_repr(minimax_model.s.data, format='{:4.3f}')))
            print('s_norm={}'.format(array1d_repr(minimax_model.get_least_s_norm(optimizer), format='{:4.3f}')))
    

    return cur_resource,minimax_model.s.data.numpy()


def build_minimax_model(net_model, layer_names, bncp_layers, bncp_layers_dict, args):
    
    print('*'*40)
    conv_layers = [m for m in net_model.modules() if isinstance(m, RCLinear)]
    skip_layers = []
    minimax_model = RC_CP_MiniMax(net_model,
                                  resource_fn = None,
                                  bncp_layers = bncp_layers,
                                  bncp_layers_dict = bncp_layers_dict, 
                                  group_size = args.group_size,
                                  punit       = args.punit)

    rc_cost_func = lambda s_, ub_: flops2(s_,
                                          bncp_layers_dict,
                                          bncp_layers, #bncp layers
                                          group_size = args.group_size,
                                          ub    = ub_,
                                          sub   = minimax_model.s_ub,
                                          layer_names = layer_names, arch = args.arch_mlp_top)

    resource_ub = float(rc_cost_func(torch.zeros_like(minimax_model.s.data), None))
    print('resource cost for full model={:.8e}'.format(resource_ub))

    # resource cost with width-multipliers:
    width_mult = [0.75, 0.5, 0.25]
    for wm in width_mult:
        r_cost = float(flops2(torch.round((1 - wm) * minimax_model.s_ub),
                                  bncp_layers_dict,
                                  bncp_layers,
                                  group_size = args.group_size,
                                  layer_names = layer_names,arch = args.arch_mlp_top)
                          )
        print('resource cost for {} model={:.8e}'.format(wm, r_cost))


    resource_fn = lambda s_: rc_cost_func(s_, resource_ub)
    minimax_model.resource_fn = resource_fn

    if args.soptim == 'adam':
        s_optimizer = torch.optim.Adam([minimax_model.s],
                                        args.slr,
                                        betas = (0.0, 0.999),
                                        weight_decay = 0.0)
    elif args.soptim == 'sgd':
        s_optimizer = torch.optim.SGD([minimax_model.s],
                                       args.slr,
                                       momentum = 0.0,
                                       weight_decay = 0.0)
    elif args.soptim == 'rmsprop':
        s_optimizer = torch.optim.RMSprop([minimax_model.s],
                                          lr = args.slr)
    else:
        raise NotImplementedError


    dual_optimizer = torch.optim.SGD([{'params': minimax_model.z, 'lr': args.zlr},
                                       {'params': minimax_model.y, 'lr': args.ylr}],
                                        1.0,
                                        momentum = 0.0,
                                        weight_decay = 0.0)

    return minimax_model, dual_optimizer, s_optimizer

