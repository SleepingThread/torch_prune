import torch
from torch import nn, tensor, bool
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn.utils import prune
from torch.nn import Conv2d, Conv1d


class VariationalDropout(object):
    def __init__(self, modules, normal_stddev=1., initial_logalpha=-12., logalpha_threshold=3., **kwargs):
        """
        We can treat VariationalDropout as hooks container
        ??: How it will be saved/loaded with torch.save/torch.load
        It should be all OK, because function/object - there is no big difference
        The other question - how pickle treat the same object <self> while loading.
        Will it create multiple VariationalDropout objects? - it's not good at all.

        Parameters
        ==========
        modules: list of (module, <dict with config>)

        Usage
        =====
        vd = VariationalDropout([(model.linear, None)])  # all specified modules support vd
        """
        self.modules = modules
        self.normal_stddev = normal_stddev
        self.initial_logalpha = initial_logalpha
        self.logalpha_threshold = logalpha_threshold

        self._modules_dict = None
        self._forward_hooks = list()
        self._forward_pre_hooks = list()

        self._build()

    def _build(self):
        """
        Add prehook and hook for all modules
        """
        self._modules_dict = dict()
        for _m, _cfg in self.modules:
            _cfg = _cfg if _cfg is not None else dict()
            self._modules_dict[_m] = _cfg
            _w_name = _cfg.get("weight", "weight")

            _w = getattr(_m, _w_name)
            delattr(_m, _w_name)
            _m.register_parameter(_w_name + "_orig", _w)
            _la = Parameter(torch.full(_w.shape, _cfg.get("init_logalpha", -12.)))
            _m.register_parameter(_w_name + "_logalpha", _la)
            _m.register_buffer(_w_name + "_mask", torch.zeros(*_w.shape, dtype=torch.bool))

            self._forward_pre_hooks.append(_m.register_forward_pre_hook(self.prehook))
            self._forward_hooks.append(_m.register_forward_hook(self.hook))

    def _base_prehook(self, module, _inputs):
        _cfg = self._modules_dict[module]
        _w_name = _cfg.get("weight", "weight")

        # calculate masked weight
        _mask = getattr(module, _w_name + "_mask")
        _la = getattr(module, _w_name + "_logalpha")
        with torch.no_grad():
            _mask[:] = _la < self.logalpha_threshold

        _weight = getattr(module, _w_name + "_orig") * _mask
        setattr(module, _w_name, _weight)

    def _base_hook(self, module, inputs, outputs):
        pass

    def _prehook_linear(self, module, inputs):
        return self._base_prehook(module, inputs)

    def _hook_linear(self, module, inputs, outputs):
        _inp = inputs[0]
        _w = module.weight
        _la = module.weight_logalpha

        _vd_add = torch.sqrt((_inp*_inp)@(torch.exp(_la)*_w*_w).t() + 1.0e-14)
        _rand = torch.normal(0., self.normal_stddev, _vd_add.shape, device=_vd_add.device)
        _vd_add = _rand*_vd_add

        return outputs + _vd_add

    def _prehook_conv2d(self, module, inputs):
        return self._base_prehook(module, inputs)

    def _hook_conv2d(self, module, inputs, outputs):
        _inp = inputs[0]
        _w = module.weight
        _la = module.weight_logalpha

        # convolve _inp*_inp with torch.exp(_la)*_w*_w, replace bias with None
        _inp = _inp*_inp
        _w = torch.exp(_la)*_w*_w
        if module.padding_mode != 'zeros':
            _vd_add = F.conv2d(F.pad(_inp, module._padding_repeated_twice, mode=module.padding_mode),
                               _w, None, module.stride,
                               torch.utils._pair(0), module.dilation, module.groups)
        else:
            _vd_add = F.conv2d(_inp, _w, None, module.stride,
                               module.padding, module.dilation, module.groups)

        _vd_add = torch.sqrt(_vd_add + 1.0e-14)
        _rand = torch.normal(0., self.normal_stddev, _vd_add.shape, device=_vd_add.device)
        _vd_add = _rand * _vd_add

        return outputs + _vd_add

    def get_dkl(self, vd_lambda):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1

        _res = 0.
        for _m, _cfg in self._modules_dict.items():
            _la = getattr(_m, _cfg.get("weight", "weight") + "_logalpha")
            mdkl = k1 * torch.sigmoid(k2 + k3 * _la) - 0.5 * torch.log1p(torch.exp(-_la)) + C
            _res += -torch.sum(mdkl)

        return vd_lambda*_res

    def remove(self):
        for _hook in self._forward_pre_hooks + self._forward_hooks:
            _hook.remove()

    def get_supported_layers(self):
        _prehook = set()
        _hook = set()
        for _el in dir(self):
            if _el.startswith("_prehook_"):
                _lr_name = _el[len("_prehook_"):]
                _prehook.add(_lr_name)
            elif _el.startswith("_hook_"):
                _lr_name = _el[len("_hook_"):]
                _hook.add(_lr_name)
        return list(_prehook.intersection(_hook))

    def prehook(self, module, input):
        _method_name = "_prehook_" + module.__class__.__name__.lower()
        return getattr(self, _method_name)(module, input)

    def hook(self, module, input, output):
        _method_name = "_hook_" + module.__class__.__name__.lower()
        return getattr(self, _method_name)(module, input, output)


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.dense1 = nn.Linear(2, 4)
        self.dense2 = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.dense2(self.dense1(inputs))

"""
- A little about torch:
    * There are tensors (torch.Tensor)
    * There are parameters (torch.nn.parameter.Parameter) - instance of torch.Tensor

- Forward-backward
    * 

- Parameter names stored aside parameters:
    * list(module.parameters())
    * list(module.named_parameters())

- Recursion inside module:
    * list(module.named_children())

- Hooks for Tensors or Modules
    * Module
        * backward_hook: (module: nn.Module, grad_input: Tensor, grad_output: Tensor) -> Tensor or None
            * 
        * forward_hook: (module: nn.Module, input: Tensor, output: Tensor) -> Tensor (modif. output) or None
        * forward_pre_hook: (module: nn.Module, input: Tensor) -> Tensor (modif. input) or None
    * Parameter
        * hook: (grad: Tensor) -> Tensor or None
        
- Manage hooks:
    * How to remove hook:
        hook = module.register_forward_hook(...)
        hook.remove()

- Hooks:
    def hook(self, input): # input hook
        pass

    def hook(self, input, output): # output hook
        pass

    module.register_forward_hook(hook)

    def back_hook(self, grad_input, grad_output):
        pass

    module.register_backward_hook(back_hook)

- Pruning and hooks:


"""