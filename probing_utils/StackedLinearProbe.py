import torch
class StackedLinearProbe(torch.nn.Module):
    def __init__(self, layers:int, in_dim:int, out_dim:int):
        super(StackedLinearProbe,self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(layers,in_dim,out_dim))
        
    def forward(self, x):
        
        Ax = torch.einsum('bplx,lxy->bply',x,self.weights)
        return torch.nn.Sigmoid()(Ax)