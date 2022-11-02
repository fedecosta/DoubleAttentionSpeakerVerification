import torch
from torch import nn


class AMSoftmax(nn.Module):

    '''
    Additve Margin Softmax as proposed in:
    https://arxiv.org/pdf/1801.05599.pdf
    Implementation Extracted From
    https://github.com/clovaai/voxceleb_trainer/blob/master/loss/cosface.py
    '''

    def __init__(self, last_features_dim, n_classes, m = 0.3, s = 15):

        super().__init__()
        self.last_features_dim = last_features_dim # dimension of the feature vector
        self.m = m
        self.s = s
        self.W = torch.nn.Parameter(torch.randn(last_features_dim, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain = 1)


    def forward(self, last_features, label = None):

        assert last_features.size()[0] == label.size()[0] # checks the batch dimension
        assert last_features.size()[1] == self.last_features_dim

        # We take the last_features vector x and connect it with a fully connected linear layer f with weights W
        # The neuron i of f is <x, W[:i]> = norm(X) * norm (W[:i]) * cos(theta)
        # We normalize x and W[:i] for each i.
        # Then, we substract the margin m only to the <x, W[:i]> corresponding to the label
        # As ouputs of this function we will have two vectors:
        # The first with all the cosines values 
        # and the second with all the cosines values except one of them with the m substraction corresponding to the label position

        # TODO torch.norm will be deprecated

        # We normalize last_features:
        # 1 - We calculate the 2_norm of last_features vector. Clamping is done to avoid division numerical problems
        x_norm = torch.norm(last_features, p = 2, dim = 1, keepdim = True).clamp(min = 1e-12) 
        # 2 - We divide by the norm
        x_norm = torch.div(last_features, x_norm)

        # We normalize the weights W over the columns (number of classes)
        w_norm = torch.norm(self.W, p = 2, dim = 0, keepdim = True).clamp(min = 1e-12)
        w_norm = torch.div(self.W, w_norm)

        # x_norm size is (1, last_features_dim) ignoring batch size
        # w_norm size is (last_features_dim, n_classes)
        # inner_products = x_norm * w_norm = (<x, W[:1]>, ..., <x, W[:n_classes]>)
        inner_products = torch.mm(x_norm, w_norm)
        
        # We need some reshaping of label
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()

        # We construct the inner_products with the corresponding -m
        aux_m = torch.zeros(inner_products.size()).scatter_(1, label_view, self.m) # make a zeros matrix with m in the corresponding label position
        if last_features.is_cuda: aux_m = aux_m.cuda() # TODO I don't know if this is ok 
        inner_products_m = inner_products - aux_m
        
        # We multiply by the scaling factor s
        inner_products_m_s = self.s * inner_products_m
        
        return inner_products, inner_products_m_s 

    
