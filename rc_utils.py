import torch
from torch import nn
from torch.nn import functional as F, Parameter
import numpy as np


class SteFloor(torch.autograd.Function):
    """
    Ste for floor function
    """
    @staticmethod
    def forward(ctx, a):
        return a.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

ste_floor = SteFloor.apply


class SteCeil(torch.autograd.Function):
    """
    Ste for ceil function
    """
    @staticmethod
    def forward(ctx, a):
        return a.ceil()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

ste_ceil = SteCeil.apply

class LeastSsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, vec):
        # vec is not sorted!
        idx = int(s.ceil().item()) + 1
        if idx <= vec.numel():
            vec_least_sp1 = torch.topk(vec, idx, largest=False, sorted=True)[0]
            ctx.vec_sp1_least = vec_least_sp1[-1].item()
            return vec_least_sp1[:-1].sum()
        else:
            ctx.vec_sp1_least = vec.max().item()
            return vec.sum()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.vec_sp1_least, None

least_s_sum = LeastSsum.apply


def flops2(s, bncp_layers_dict, process_layers,group_size, ub=None, sub=None, layer_names=None,arch="512-256-1"):
    res = 0.0

    if len(process_layers) == 1:
        layer = process_layers[0][0]
        in_dim = float(layer.in_features)
        out_dim = float(layer.out_features)
        in_s =  s[bncp_layers_dict[layer]]

        res += 2 * ste_floor(in_dim - in_s * group_size) * out_dim  + out_dim

        arch_array = np.fromstring(arch, dtype=int, sep="-")
        for i in range(arch_array.shape[0] - 1):
            res += 2 * arch_array[i] * arch_array[i+1] + arch_array[i+1]




    elif len(process_layers) > 1:
        for layer_group in process_layers:
            for layer in layer_group:
                in_dim = float(layer.in_features)
                out_dim = float(layer.out_features)
                in_s =  s[bncp_layers_dict[layer]]

                if bncp_layers_dict[layer] < len(process_layers)-1:
                    out_s = s[bncp_layers_dict[layer]+1]
                    assert in_s <= in_dim and out_s <= out_dim


                if bncp_layers_dict[layer] == 0: # input layer
                    res += 2 * ste_floor(in_dim - in_s * group_size) * ste_floor(out_dim - out_s) + ste_floor(out_dim - out_s)
                elif bncp_layers_dict[layer] < len(process_layers) - 1:
                    res += 2* ste_floor(in_dim - in_s)*ste_floor(out_dim - out_s) + ste_floor(out_dim - out_s)
                else:
                    res += 2 * ste_floor(in_dim - in_s) * out_dim + out_dim



    return res / ub if ub is not None else res



def weight_list_to_scores(weight_list,layers_dict,group_size = 1, eps=None, optimizer=None):
    # input: weight matrix (a list containing one element )
    # output: L2 norm of the group tensors of the weight matrix (shape: the length of total groups of variables)
    eps = None
    if eps is not None:
        return sum([optimizer.state[m.weight]['exp_avg_sq'].data.sqrt().add_(eps) * (m.weight.data ** 2)
                    for m in weight_list])
    else:
        result = []
        for m in weight_list:
            if layers_dict[m] != 0: # hidden layers
                result.append((m.weight.data ** 2).sum(0))
            else:  # input layer
                n_group = m.weight.data.shape[1] // group_size
                for index_group in range(n_group):
                    group_weight = m.weight.data[:,index_group * group_size : (index_group + 1) * group_size]
                    group_weight = group_weight.reshape(-1)
                    result.append((group_weight **2).sum(0))
                result = [torch.tensor(result)]

        res = sum(result) 
        return res

class RC_CP_MiniMax(nn.Module):
    """
    Resource-Constrained channel pruning minimax:
    min_{w, s} max_{y>=0, z>=0} L(w) + <y, |w|_{ceil(s), 2}^2> + z*(resource(s) - B)
    """
    def __init__(self, net_model, resource_fn, bncp_layers, bncp_layers_dict, group_size, z_init=0.0, y_init=0.0, punit=1):
        super(RC_CP_MiniMax, self).__init__()
        self.net = net_model
        self.bncp_layers = bncp_layers
        self.bncp_layers_dict = bncp_layers_dict
        self.group_size = group_size
        n_layers = len(self.bncp_layers)
        self.s = Parameter(torch.zeros(n_layers))
        self.y = Parameter(torch.Tensor(n_layers))
        self.y.data.fill_(y_init)
        self.z = Parameter(torch.tensor(float(z_init)))
        self.resource_fn = resource_fn
        self.__least_s_norm = torch.zeros_like(self.s.data)
        self.s_ub = torch.zeros_like(self.s.data)
        assert punit >= 1
        self.punit = float(punit)
        # print("test3", self.bncp_layers)
        for i, layers in enumerate(self.bncp_layers):
            for layer in layers:
                assert layer.weight.numel() == layers[0].weight.numel()
            if self.bncp_layers_dict[layer] == 0:
                self.s_ub[i] = layers[0].in_features // self.group_size
            else:
                self.s_ub[i] = layers[0].in_features

    def ceiled_s(self):
        if self.punit > 1.0:
            return torch.cat((ste_ceil(self.s[0].view(-1)), ste_ceil(self.s[1:self.s.shape[0]] / self.punit) * self.punit)) 
        else:
            return ste_ceil(self.s)

    def zloss(self, budget):
        return self.z * (self.resource_fn(self.ceiled_s().data) - budget)

    def sloss1(self, optimizer):
        if isinstance(optimizer, torch.optim.Adam):
            eps = optimizer.param_groups[0]['eps']
        else:
            eps = None
        s = self.ceiled_s()

        w_s_norm = torch.empty(1)
        for i, layers in enumerate(self.bncp_layers):
            temp = least_s_sum(s[i],weight_list_to_scores(layers, self.bncp_layers_dict, self.group_size, eps=eps).data.cpu()).view(-1)
            w_s_norm = torch.cat((w_s_norm, temp))

        w_s_norm = w_s_norm[1:w_s_norm.shape[0]]

        return self.y.data.dot(w_s_norm)

    def sloss2(self, budget):
        s = self.ceiled_s()
        rc = self.resource_fn(s)
        return rc - budget

    def get_least_s_norm(self, optimizer):
        if isinstance(optimizer, torch.optim.Adam):
            eps = optimizer.param_groups[0]['eps']
        else:
            eps = None
        res = self.__least_s_norm
        s = self.ceiled_s()
        for i, layers in enumerate(self.bncp_layers):
            scores = weight_list_to_scores(layers, self.bncp_layers_dict, self.group_size, eps=eps)
            res[i] = torch.topk(scores, int(s[i].ceil().item()), largest=False, sorted=False)[0].sum().item()
        return res

    def yloss(self, w_optimizer):
        return self.y.dot(self.get_least_s_norm(w_optimizer))

def prox_w(minimax_model, optimizer):
    lr = optimizer.param_groups[0]['lr']
    if isinstance(optimizer, torch.optim.Adam):
        eps = optimizer.param_groups[0]['eps']
    else:
        eps = None

    s = minimax_model.ceiled_s()
    
    for i, layers in enumerate(minimax_model.bncp_layers):
        scores = weight_list_to_scores(layers,minimax_model.bncp_layers_dict, minimax_model.group_size,eps=eps)

        least_s_idx = torch.topk(scores, int(s[i].ceil().item()), largest=False, sorted=False)[1]  

        for m in layers:
            if minimax_model.bncp_layers_dict[m] == 0:
                for index_group in least_s_idx:
                    m.weight.data[:, index_group * minimax_model.group_size :  (index_group + 1) * minimax_model.group_size]  /= (1.0 + 2.0 * lr * minimax_model.y[i].item())

            else:
                m.weight.data[:,least_s_idx] /= (1.0 + 2.0 * lr * minimax_model.y[i].item())

def make_zero_w(minimax_model, optimizer):
    lr = optimizer.param_groups[0]['lr']
    if isinstance(optimizer, torch.optim.Adam):
        eps = optimizer.param_groups[0]['eps']
    else:
        eps = None

    s = minimax_model.ceiled_s()
    least_s_idx_all=[]
#########------------------------

    for i, layers in enumerate(minimax_model.bncp_layers):
        scores = weight_list_to_scores(layers,minimax_model.bncp_layers_dict, minimax_model.group_size,eps=eps)

        least_s_idx = torch.topk(scores, int(s[i].ceil().item()), largest=False, sorted=False)[1] 
        least_s_idx_all.append(least_s_idx)

    for i, layers in enumerate(minimax_model.bncp_layers):
        for m in layers:
            print("m",m)
            if minimax_model.bncp_layers_dict[m] == 0:
                for index_group in least_s_idx_all[0]:
                    m.weight.data[:, index_group * minimax_model.group_size :  (index_group + 1) * minimax_model.group_size] = 0
                if len(least_s_idx_all) > 1:    
                    m.weight.data[least_s_idx_all[1],:] = 0

            elif minimax_model.bncp_layers_dict[m] < len(minimax_model.bncp_layers) - 1:
                cur_dict = minimax_model.bncp_layers_dict[m]
                m.weight.data[:,least_s_idx_all[cur_dict]] = 0
                m.weight.data[least_s_idx_all[cur_dict+1],:] = 0

            elif minimax_model.bncp_layers_dict[m] == len(minimax_model.bncp_layers) - 1:
                cur_dict = minimax_model.bncp_layers_dict[m]
                m.weight.data[:,least_s_idx_all[cur_dict]] = 0

    

def proj_dual(minimax_model):
    minimax_model.y.data.clamp_(min=0.0)
    minimax_model.z.data.clamp_(min=0.0)
    # ensure that y and z  be positive

