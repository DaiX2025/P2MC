import torch.nn as nn

num_parallel = 4

class ModuleParallel(nn.Module):
    
    def __init__(self, module, module_name):
        super(ModuleParallel, self).__init__()
        self.module = module
        self.module_name = module_name
        for i in range(num_parallel):
            setattr(self, str(module_name) + "_" + str(i), module)

    def forward(self, x_parallel):
        
        return [getattr(self, str(self.module_name) + "_" + str(i))(x) for i, x in enumerate(x_parallel)]
