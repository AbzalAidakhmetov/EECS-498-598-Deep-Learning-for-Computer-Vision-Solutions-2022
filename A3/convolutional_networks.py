"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        pad = conv_param['pad']
        stride = conv_param['stride']

        x_padded = torch.nn.functional.pad(x, pad=(pad, pad, pad, pad)) # C, H_pad, W_pad

        N, C, H_orig, W_orig = x.shape
        N, C, H, W = x_padded.shape
        F, C, HH, WW = w.shape
        H_new = int(1 + (H_orig + 2 * pad - HH) / stride)
        W_new = int(1 + (W_orig + 2 * pad - WW) / stride)

        # Note: You have to make the same dtype as x which is torch.float64, by default torch.zeros will create torch.float32 dtype, which will cause error
        # when calculating
        # out = torch.zeros(size=(N, F, H_new, W_new))
        # print(x.dtype, torch.zeros(size=(N, F, H_new, W_new)).dtype)
        out = torch.zeros(size=(N, F, H_new, W_new), dtype=x.dtype)

        for n in range(N):
            cur_sample = x_padded[n] # C, H, W

            for f in range(F):
                cur_filter = w[f] # C, HH, WW

                channel, height, width = cur_sample.shape
                for j in range(0, height, stride):
                      for i in range(0, width, stride):
                          if i + WW <= W and j + HH <= H:
                              out[n][f][j//stride][i//stride] = torch.sum(cur_sample[:, j: j + HH, i: i + WW] * cur_filter)
                          else:
                              continue
            
        out = out + b[None, :, None, None]            
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        x, w, b, conv_param = cache
        pad = conv_param['pad']
        stride = conv_param['stride']

        x_padded = torch.nn.functional.pad(x, pad=(pad, pad, pad, pad)) # C, H_pad, W_pad

        N, C, H_orig, W_orig = x.shape
        # N, C, H, W = x_padded.shape
        N, F, H_new, W_new = dout.shape
        F, C, HH, WW = w.shape
        dx_padded = torch.zeros(*x_padded.shape, dtype=torch.float64)
        # dx_padded = torch.zeros_like(x)
        dw = torch.zeros(*w.shape, dtype=w.dtype)
        db = torch.zeros(*b.shape, dtype=b.dtype)
       
        for n in range(N):
            for f in range(F):
              for height in range(H_new):
                  for width in range(W_new):
                    
                        dx_padded[n, :, height * stride: height * stride + HH, width * stride: width * stride + WW] += w[f] * dout[n, f, height, width]
                        db[f] += dout[n, f, height, width]
                        dw[f] += x_padded[n, :, height * stride: height * stride + HH, width * stride: width * stride + WW] * dout[n, f, height, width]

        
        dx = dx_padded[:, :, 1: -1, 1: -1]

        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        N, C, H, W = x.shape
        pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        H_new = int(1 + (H - pool_height) / stride)
        W_new = int(1 + (W - pool_width) / stride)

        out = torch.zeros(size=(N, C, H_new, W_new), dtype=x.dtype)

        for n in range(N):
            for c in range(C):
                for h in range(H_new):
                    for w in range(W_new):
                        out[n, c, h, w] = torch.max(x[n, c, h * stride: h * stride + pool_height, w * stride: w * stride + pool_width])
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        x, pool_param = cache
        dx = torch.zeros_like(x)
        
        N, C, H, W = x.shape
        pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        H_new = int(1 + (H - pool_height) / stride)
        W_new = int(1 + (W - pool_width) / stride)

        for n in range(N):
            for c in range(C):
                for h in range(H_new):
                    for w in range(W_new):
                        cur_max_val = torch.max(x[n, c, h * stride: h * stride + pool_height, w * stride: w * stride + pool_width])
                        dx[n, c, h * stride: h * stride + pool_height, w * stride: w * stride + pool_width][[x[n, c, h * stride: h * stride + pool_height, w * stride: w * stride + pool_width] == cur_max_val]] = 1
                        dx[n, c, h * stride: h * stride + pool_height, w * stride: w * stride + pool_width] *= dout[n, c, h, w]

        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        C, H, W = input_dims
        self.params['W1'] = torch.randn(size=(num_filters, C, filter_size, filter_size), dtype=dtype, device=device) * weight_scale
        self.params['b1'] = torch.zeros(size=(num_filters,), dtype=dtype, device=device)
        self.params['W2'] = torch.randn(size=(num_filters * (H // 2) * (W // 2), hidden_dim), dtype=dtype, device=device) * weight_scale
        self.params['b2'] = torch.zeros(size=(hidden_dim,), dtype=dtype, device=device)
        self.params['W3'] = torch.randn(size=(hidden_dim, num_classes), dtype=dtype, device=device) * weight_scale
        self.params['b3'] = torch.zeros(size=(num_classes,), dtype=dtype, device=device)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        # Replace "pass" statement with your code
        crp_out, crp_cache = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)

        lrelu_out, lrelu_cache = Linear_ReLU.forward(crp_out, W2, b2)
        linear_out, linear_cache = Linear.forward(lrelu_out, W3, b3)
        scores = linear_out
        loss_softmax, grad_softmax = softmax_loss(linear_out, y)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # Replace "pass" statement with your code
        loss = loss_softmax + self.reg * torch.sum(W1 * W1) + self.reg * torch.sum(W2 * W2) + self.reg * torch.sum(W3 * W3)
        upstream, dW3, db3 = Linear.backward(grad_softmax, linear_cache)
        grads['W3'] = dW3 + 2 * self.reg * W3
        grads['b3'] = db3
        upstream, dW2, db2 = Linear_ReLU.backward(upstream, lrelu_cache)
        grads['W2'] = dW2 + 2 * self.reg * W2
        grads['b2'] = db2
        upstream, dW1, db1 = Conv_ReLU_Pool.backward(upstream, crp_cache)
        grads['W1'] = dW1 + 2 * self.reg * W1
        grads['b1'] = db1
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        # Replace "pass" statement with your code
        

        C, H, W = input_dims
        cur_H = H
        cur_W = W

        if weight_scale == 'kaiming':
            for l in range(self.num_layers - 1):
                if l == 0:
                    self.params[f'W_{l + 1}'] = kaiming_initializer(Din = C, Dout = num_filters[l], K = 3, dtype=dtype, device=device)
                else:
                    self.params[f'W_{l + 1}'] = kaiming_initializer(Din = num_filters[l - 1], Dout = num_filters[l], K = 3, dtype=dtype, device=device)

                self.params[f'b_{l + 1}'] = torch.zeros(size=(num_filters[l],), dtype=dtype, device=device)
                if self.batchnorm:
                    self.params[f'gamma_{l + 1}'] = torch.ones(size=(num_filters[l], ), dtype=dtype, device=device)
                    self.params[f'beta_{l + 1}'] = torch.zeros(size=(num_filters[l], ), dtype=dtype, device=device)
                if l in self.max_pools:
                    cur_H //= 2
                    cur_W //= 2

    
            self.params[f'W_{self.num_layers}'] = kaiming_initializer(Dout=num_classes, Din = num_filters[-1] * cur_H * cur_W, relu=False, dtype=dtype, device=device)
            self.params[f'b_{self.num_layers}'] = torch.zeros(size=(num_classes,), dtype=dtype, device=device)
        else:
            for l in range(self.num_layers - 1):
                if l == 0:
                    self.params[f'W_{l + 1}'] = torch.randn(size=(num_filters[l], C, 3, 3), dtype=dtype, device=device) * weight_scale
                else:
                    self.params[f'W_{l + 1}'] = torch.randn(size=(num_filters[l], num_filters[l - 1], 3, 3), dtype=dtype, device=device) * weight_scale

                self.params[f'b_{l + 1}'] = torch.zeros(size=(num_filters[l],), dtype=dtype, device=device)
                if self.batchnorm:
                    self.params[f'gamma_{l + 1}'] = torch.ones(size=(num_filters[l], ), dtype=dtype, device=device)
                    self.params[f'beta_{l + 1}'] = torch.zeros(size=(num_filters[l], ), dtype=dtype, device=device)
                if l in self.max_pools:
                    cur_H //= 2
                    cur_W //= 2

    
            self.params[f'W_{self.num_layers}'] = torch.randn(size=(num_filters[-1] * cur_H * cur_W, num_classes), dtype=dtype, device=device) * weight_scale
            self.params[f'b_{self.num_layers}'] = torch.zeros(size=(num_classes,), dtype=dtype, device=device)
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code
        cache = {}

        for l in range(self.num_layers - 1):
            
            if l == 0:
                if l in self.max_pools:
                    if self.batchnorm:
                        cache[f'out_{l + 1}'], cache[f'cache_{l + 1}'] = Conv_BatchNorm_ReLU_Pool.forward(X, self.params[f'W_{l + 1}'], self.params[f'b_{l + 1}'], self.params[f'gamma_{l + 1}'], self.params[f'beta_{l + 1}'], conv_param, {'mode': 'train'}, pool_param)                       
                    else:
                        cache[f'out_{l + 1}'], cache[f'cache_{l + 1}'] = Conv_ReLU_Pool.forward(X, self.params[f'W_{l + 1}'], self.params[f'b_{l + 1}'], conv_param, pool_param)
                else:
                    if self.batchnorm:
                        cache[f'out_{l + 1}'], cache[f'cache_{l + 1}'] = Conv_BatchNorm_ReLU.forward(X, self.params[f'W_{l + 1}'], self.params[f'b_{l + 1}'], self.params[f'gamma_{l + 1}'], self.params[f'beta_{l + 1}'], conv_param,  {'mode': 'train'})
                    else:
                        cache[f'out_{l + 1}'], cache[f'cache_{l + 1}'] = Conv_ReLU.forward(X, self.params[f'W_{l + 1}'], self.params[f'b_{l + 1}'], conv_param)

                continue
            
            if l in self.max_pools:
                if self.batchnorm:
                    cache[f'out_{l + 1}'], cache[f'cache_{l + 1}'] = Conv_BatchNorm_ReLU_Pool.forward(cache[f'out_{l}'], self.params[f'W_{l + 1}'], self.params[f'b_{l + 1}'], self.params[f'gamma_{l + 1}'], self.params[f'beta_{l + 1}'], conv_param, {'mode': 'train'}, pool_param)
                else:
                    cache[f'out_{l + 1}'], cache[f'cache_{l + 1}'] = Conv_ReLU_Pool.forward(cache[f'out_{l}'], self.params[f'W_{l + 1}'], self.params[f'b_{l + 1}'], conv_param, pool_param)
            else:
                if self.batchnorm:
                    cache[f'out_{l + 1}'], cache[f'cache_{l + 1}'] = Conv_BatchNorm_ReLU.forward(cache[f'out_{l}'], self.params[f'W_{l + 1}'], self.params[f'b_{l + 1}'], self.params[f'gamma_{l + 1}'], self.params[f'beta_{l + 1}'], conv_param,  {'mode': 'train'})
                else:
                    cache[f'out_{l + 1}'], cache[f'cache_{l + 1}'] = Conv_ReLU.forward(cache[f'out_{l}'], self.params[f'W_{l + 1}'], self.params[f'b_{l + 1}'], conv_param)

        
        cache[f'out_{self.num_layers}'], cache[f'cache_{self.num_layers}'] = Linear.forward(cache[f'out_{self.num_layers - 1}'], self.params[f'W_{self.num_layers}'], self.params[f'b_{self.num_layers}'])
        scores = cache[f'out_{self.num_layers}']
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code
        loss_softmax, grad_softmax = softmax_loss(cache[f'out_{self.num_layers}'], y)
        loss = loss_softmax
        for l in range(self.num_layers):
            loss += self.reg * torch.sum(self.params[f'W_{l + 1}'] ** 2)

        upstream, dW_last, db_last = Linear.backward(grad_softmax, cache[f'cache_{self.num_layers}'])
        grads[f'W_{self.num_layers}'] = dW_last + 2 * self.reg * self.params[f'W_{self.num_layers}']
        grads[f'b_{self.num_layers}'] = db_last
        
        for l in range(self.num_layers - 1, 0, -1):
            # print(l)
            if l - 1 in self.max_pools:
                if self.batchnorm:
                    upstream, dW, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(upstream, cache[f'cache_{l}'])
                    grads[f'gamma_{l}'] = dgamma 
                    grads[f'beta_{l}'] = dbeta
                else:
                    upstream, dW, db = Conv_ReLU_Pool.backward(upstream, cache[f'cache_{l}'])
            else:
                if self.batchnorm:
                    upstream, dW, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(upstream, cache[f'cache_{l}'])
                    grads[f'gamma_{l}'] = dgamma 
                    grads[f'beta_{l}'] = dbeta  
                else:
                    upstream, dW, db = Conv_ReLU.backward(upstream, cache[f'cache_{l}'])

            grads[f'W_{l}'] = dW + 2 * self.reg * self.params[f'W_{l}']
            grads[f'b_{l}'] = db

        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = (2 / (3 * 3 * 3))   # Experiment with this!
    learning_rate = 1e-3  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    pass
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # Train the best DeepConvNet that you can on CIFAR-10   #
    # within the existing code structure. Only parameters   #
    # are changed below.                                    #
    #########################################################
    # (Same structure, just tweaked arguments.)
    
    input_dims = data_dict['X_train'].shape[1:]
    model = DeepConvNet(
        input_dims=input_dims,
        num_classes=10,
        num_filters=[32, 64, 128, 128],  
        max_pools=[0, 1, 2, 3],         
        weight_scale='kaiming',        
        reg=5e-5,                      
        dtype=torch.float32,
        device='cpu'
    )

    solver = Solver(
        model, data_dict,
        num_epochs=25,       
        batch_size=128,      
        update_rule=adam,
        optim_config={
            'learning_rate': 1e-3,   
        },
        print_every=20,
        device='cpu'
    )
    #########################################################
    #                  END OF PARAMETER TWEAKS              #
    #########################################################
    return solver



def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        weight = torch.randn(size=(Din, Dout), dtype=dtype, device=device) * torch.tensor((gain / Din) ** 0.5, dtype=dtype, device=device)
        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        weight = torch.randn(size=(Dout, Din, K, K), dtype=dtype, device=device) * torch.tensor((gain / ((K ** 2) * Din)) ** 0.5, dtype=dtype, device=device)
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code
            sample_mean = torch.mean(x, dim=0)

            # Note: important to do unbiased to False, otherwise you will be dividing by (N - 1), instead of paper's N
            sample_var = torch.var(x, dim=0, unbiased=False)

            normalized = (x - sample_mean) / torch.sqrt(sample_var + eps)
            out = gamma * normalized + beta

            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var

            cache = {'sample_mean': sample_mean, 'sample_var': sample_var, 'gamma': gamma, 'beta': beta, 'normalized': normalized, 'x': x, 'eps':eps}
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code
            normalized = (x - running_mean) / torch.sqrt(running_var + eps)
            out = gamma * normalized + beta
            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your code
        

        sample_mean = cache['sample_mean']
        sample_var = cache['sample_var']
        gamma = cache['gamma']
        beta = cache['beta']
        normalized = cache['normalized']
        x = cache['x']
        eps = cache['eps']
        N, D = x.shape

        grad_dx_hat = dout * gamma

        grad_var = (grad_dx_hat * (x - sample_mean) * (-1/2) * (sample_var + eps) ** (-3/2)).sum(dim=0)
        
        grad_mean = (grad_dx_hat * -1 / (sample_var + eps) ** 0.5).sum(dim=0) + grad_var * (-2 * (x - sample_mean)).sum(dim=0) / N
        
        dx = grad_dx_hat / (sample_var + eps) ** 0.5 + grad_var * 2 * (x - sample_mean) / N  + grad_mean / N


        dgamma = torch.sum(dout * normalized, dim=0)
        dbeta = torch.sum(dout, dim=0)

        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    # Note: CHATGPT's answer for alternative one
    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        Implements a simplified expression that fits (almost)
        in one or two lines of code for dx.
        """
        # Unpack from cache what we need
        x_hat = cache['normalized']           # shape (N, D)
        gamma = cache['gamma']               # shape (D,)
        sample_var = cache['sample_var']     # shape (D,)
        eps = cache['eps']
        x = cache['x']
        N, D = x.shape

        # 1) dgamma, dbeta are the same as usual
        dgamma = torch.sum(dout * x_hat, dim=0)  # shape (D,)
        dbeta  = torch.sum(dout, dim=0)          # shape (D,)

        # 2) dx_hat = dout * gamma
        dx_hat = dout * gamma  # shape (N, D)

        # 3) Summations needed for the simplified formula
        sum_dx_hat        = torch.sum(dx_hat, dim=0)               # shape (D,)
        sum_dx_hat_x_hat  = torch.sum(dx_hat * x_hat, dim=0)       # shape (D,)

        # 4) One-liner (or two-liner) to compute dx
        dx = (1.0 / (N * torch.sqrt(sample_var + eps))) * (
                (N * dx_hat)
                - sum_dx_hat
                - x_hat * sum_dx_hat_x_hat
            )

        return dx, dgamma, dbeta



class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code

        N, C, H, W = x.shape
        out, cache = BatchNorm.forward(x.permute(0, 2, 3, 1).reshape(-1, C), gamma, beta, {'mode': 'train'})
        out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)

        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code
        N, C, H, W = dout.shape
        dx, dgamma, dbeta = BatchNorm.backward(dout.permute(0, 2, 3, 1).reshape(-1, C), cache)
        dx = dx.reshape(N, H, W, C).permute(0, 3, 1, 2)
        
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
