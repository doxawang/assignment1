import numpy as np
from math import *


class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.
            
            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object, 
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_byconst_op(self, other)
        return new_node
        """TODO: Your code here"""
    
    def sin(self):
        new_node = sin_op(self)
        return new_node

    def cos(self):
        new_node = cos_op(self)
        return new_node

    def tan(self):
        new_node = tan_op(self)
        return new_node

    def log(self):
        new_node = log_op(self)
        return new_node
    
    def exp(self):
        new_node = exp_op(self)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name.""" 
        return self.name

    __repr__ = __str__

def Variable(name):
    """User defined variables in an expression.  
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node

class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.
        
        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

class AddOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [output_grad, output_grad]

class AddByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [output_grad]

class MulOp(Op):
    """Op to element-wise multiply two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]
        """TODO: Your code here"""

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input."""
        return [mul_op(output_grad,node.inputs[1]), mul_op(output_grad, node.inputs[0])]
        """TODO: Your code here"""

class MulByConstOp(Op):
    """Op to element-wise multiply a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr
        """TODO: Your code here"""

    def gradient(self, node, output_grad):
        """Given gradient of multiplication node, return gradient contribution to input."""
        return [mul_byconst_op(output_grad, node.const_attr)]
        """TODO: Your code here"""

class MatMulOp(Op):
    """Op to matrix multiply two nodes."""
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        node.input_vals = input_vals
        i, j = input_vals[0].shape
        k, l = input_vals[1].shape
        if node.matmul_attr_trans_A == 0:
            if node.matmul_attr_trans_B == 0:
                assert j == k
                return np.matmul(input_vals[0], input_vals[1])
            else:
                assert j == l
                return np.matmul(input_vals[0], input_vals[1].T)
        if node.matmul_attr_trans_A == 1:
            if node.matmul_attr_trans_B == 0:
                assert i == k
                return np.matmul(input_vals[0].T, input_vals[1])
            else:
                assert i == l
                return np.matmul(input_vals[0].T, input_vals[1].T)
        """TODO: Your code here"""

    def gradient(self, node, output_grad):  ###???
        """Given gradient of multiply node, return gradient contributions to each input.
            
        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        if node.matmul_attr_trans_A == 0:
            if node.matmul_attr_trans_B == 0:
                return [matmul_op(output_grad, node.inputs[1], trans_A=0, trans_B=1),
                        matmul_op(node.inputs[0], output_grad, trans_A=1, trans_B=0)]
        if node.matmul_attr_trans_A == 0:
            if node.matmul_attr_trans_B == 1:
                return [matmul_op(output_grad, node.inputs[1], trans_A=0, trans_B=0),
                        matmul_op(output_grad, node.inputs[1], trans_A=1, trans_B=0)]
        if node.matmul_attr_trans_A == 1:
            if node.matmul_attr_trans_B == 0:
                return [matmul_op(node.inputs[1], output_grad, trans_A=0, trans_B=1),
                        matmul_op(node.inputs[0], output_grad, trans_A=0, trans_B=0)]
        if node.matmul_attr_trans_A == 1:
            if node.matmul_attr_trans_B == 1:
                return [matmul_op(node.inputs[1], output_grad, trans_A=1, trans_B=1),
                        matmul_op(output_grad, node.inputs[0], trans_A=1, trans_B=1)]
        """TODO: Your code here"""

class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None

class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class SinOp(Op):
    
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "sin(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 1)
        return np.sin(input_vals[0])

    def gradient(self, node, output_grad):
        return [mul_op(output_grad, cos_op(node.inputs[0]))]


class CosOp(Op):

    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "cos(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 1)
        return np.cos(input_vals[0])

    def gradient(self, node, output_grad):
        return [mul_byconst_op(mul_op(output_grad, sin_op(node.inputs[0])), -1)]

class ExpOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "exp(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 1)
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [mul_op(output_grad, exp_op(node.inputs[0]))]

class InvOp(Op): 
    """Point-wise inverse operation of the input node"""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "inv(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 1)
        assert(input_vals != 0)
        return 1/input_vals[0]

    def gradient(self, node, output_grad):
        return [mul_op(output_grad, 
                inv_op(mul_op(node.inputs[0], node.inputs[0])))]

class LogOp(Op): 
    """Point-wise log operation of the input node"""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "log(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 1)
        assert(input_vals != 0)
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [mul_op(output_grad, inv_op(node.inputs[0]))]

class PowerOp(Op):
    """Point-wise power operation of the input node"""
    def __call__(self, node_A, const):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = const
        new_node.name = "power(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 1)
        return np.power(input_vals[0], node.const_attr)
 
    def gradient(self, node, output_grad):
        return [mul_op(output_grad, mul_byconst_op(node.const_attr, power_op(node.inputs[0], node.const_attr-1)))]

class TanOp(Op):
    """Point-wise tan operation of the input node"""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "tan(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 1)
        assert(input_vals != 0)
        return np.tan(input_vals[0])

    def gradient(self, node, output_grad):
        return [mul_op(output_grad, power_op(cos_op(node.inputs[0]),-2))]
        # return [inv_op(mul_op(cos_op(node.inputs[0]),cos_op(node.inputs[0])))]


class SigmoidOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "sigmoid(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 1)
        return 1/(1+np.exp(-input_vals[0]))

    def gradient(self, node, output_grad):
        return [mul_op(sigmoid_op(node.inputs[0]), 
                        add_byconst_op(mul_byconst_op(node.inputs[0],-1),1))]

class SignOp(Op):
    """(sign(X))_ij = 1 if X_ij>0 else 0 """
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "sign(%s)" % node_A.name 
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 1)
        return input_vals[0] > 0

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class ReLuOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "relu(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 1)
        return np.max(input_vals[0], 0)

    def gradient(self, node, output_grad):
        return mul_op(output_grad, sign_op(node.inputs[0]))

class ConvOp(Op):

    """ Assume no padding and stride = 1
    Reference: https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199 
    """

    def __call__(self, node_X, node_W):        
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, node_W]
        new_node.name = "conv(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        (X, W) = input_vals

        # Retrieving dimensions from X's shape
        (n_H_prev, n_W_prev) = X.shape
        (f, f) = W.shape

        # Retrieving dimensions from W's shape
        (f, f) = W.shape        
        
        n_H = n_H_prev - f + 1
        n_W = n_W_prev - f + 1
        
        # Initialize the output H with zeros
        H = np.zeros((n_H, n_W))
        
        # Looping over vertical(h) and horizontal(w) axis of output volume
        for h in range(n_H):
            for w in range(n_W):
                x_slice = X[h:h+f, w:w+f]
                H[h,w] = np.sum(x_slice * W)

        return H

        # Saving information in 'cache' for backprop
        cache = (X, W)

    def gradient(self, node, output_grad):
        return [mul_op(output_grad, conv_grad_x_op(node.inputs[0], node.inputs[1])),
                mul_op(output_grad, conv_grad_w_op(node.inputs[0], node.inputs[1]))]


class ConvGradXOp(Op):
    def __call__(self, node_X, node_W):
        grad_X = Op.__call__(self)
        grad_W = Op.__call__(self)
        grad_X.inputs = [node_A, node_W]
        grad_X.name = "conv_grad_x(%s,%s)" % (node_X.name, node_W.name)
        return grad_X

    def compute(self, node, input_vals):
        (X, W, output_grad) = input_vals
        # Retrieving dimensions from X's shape
        (n_H_prev, n_W_prev) = X.shape
        
        # Retrieving dimensions from W's shape
        (f, f) = W.shape
        
        # Initializing dX, dW with the correct shapes
        dX = np.zeros(X.shape)
        dW = np.zeros(W.shape)

        n_H = n_H_prev - f + 1
        n_W = n_W_prev - f + 1

        # Looping over vertical(h) and horizontal(w) axis of the output
        for h in range(n_H):
            for w in range(n_W):
                dX[h:h+f, w:w+f] += W
        
        return dX

    def gradient(self, node, output_grad):
        return None 


class ConvGradWOp(Op):
    def __call__(self, node_X, node_W):
        grad_W = Op.__call__(self)
        grad_W.inputs = [node_A, node_W]
        grad_W.name = "conv_grad_w(%s,%s)" % (node_X.name, node_W.name)  
        return grad_W

    def compute(self, node, input_vals):
        (X, W, output_grad) = input_vals
        # Retrieving dimensions from X's shape
        (n_H_prev, n_W_prev) = X.shape
        
        # Retrieving dimensions from W's shape
        (f, f) = W.shape
        
        # Initializing dX, dW with the correct shapes
        dX = np.zeros(X.shape)
        dW = np.zeros(W.shape)

        n_H = n_H_prev - f + 1
        n_W = n_W_prev - f + 1
            
        # Looping over vertical(h) and horizontal(w) axis of the output
        for h in range(n_H):
            for w in range(n_W):
                dW += X[h:h+f, w:w+f] * output_grad(h,w)

        return dW

    def gradient(self, node, output_grad):
        return None 



class BatchNormOp(Op):
    """ 
    Reference: http://cthorey.github.io/backpropagation/
    """

    def __call__(self, node_X, node_gamma, node_beta, eps=1e-3):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, node_gamma, node_beta]
        new_node.name = "batch_norm(%s)" % node_A.name
        new_node.eps = eps
        return new_node

    def compute(self, node, input_vals):
        assert(len(input_vals) == 3)
        (X, gamma, beta) = input_vals
        epsilon = node.eps
        mu = 1/N*np.sum(X,axis =0) # Size (X,) 
        sigma2 = 1/N*np.sum((h-mu)**2,axis=0) # Size (X,) 
        hatX = (X-mu)*(sigma2+epsilon)**(-1./2.)
        return hatX


    def gradient(self, node, output_grad):
        return [batch_norm_grad_x_op(node.inputs[0], node.inputs[1], node.inputs[2], output_grad, node.eps), 
                batch_norm_grad_gamma_op(node.inputs[0], node.inputs[1], node.inputs[2], output_grad, node.eps), 
                batch_norm_grad_beta_op(node.inputs[0], node.inputs[1], node.inputs[2], output_grad, node.eps)]


class BatchNormGradXOp(Op):
    def __call__(self, node_X, node_gamma, node_beta, output_grad, eps):
        grad_X = Op.__call__(self)
        grad_X.inputs = [node_X, node_gamma, node_beta, output_grad]
        grad_X.name = "batch_norm_grad_x(%s, %s, %s)" % (node_X, node_gamma, node_beta)  
        self.eps = eps
        return grad_X

    def compute(self, node, input_vals):
        (X, gamma, beta, output_grad) = input_vals
        eps = self.eps
        # Retrieving dimensions from X's shape
        mu = 1./N*np.sum(X, axis = 0)
        var = 1./N*np.sum((X-mu)**2, axis = 0)
        dX = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * output_grad - np.sum(output_grad, axis=0)
                - (X - mu) * (var + eps)**(-1.0) * np.sum(output_grad * (X - mu), axis=0))
        return dX


    def gradient(self, node, output_grad):
        return None 


class BatchNormGradGammaOp(Op):
    def __call__(self, node_X, node_gamma, node_beta, output_grad):
        grad_gamma = Op.__call__(self)
        grad_gamma.inputs = [node_X, node_gamma, node_beta, output_grad]
        grad_gamma.name = "batch_norm_grad_gamma(%s, %s, %s)" % (node_X, node_gamma, node_beta)  
        self.eps = eps        
        return grad_gamma

    def compute(self, node, input_vals):
        (X, gamma, beta, output_grad) = input_vals
        eps = self.eps        
        # Retrieving dimensions from X's shape
        mu = 1./N*np.sum(X, axis = 0)
        var = 1./N*np.sum((X-mu)**2, axis = 0)
        dgamma = np.sum((X - mu) * (var + eps)**(-1. / 2.) * output_grad, axis=0)

        return dgamma


    def gradient(self, node, output_grad):
        return None 

class BatchNormGradBetaOp(Op):
    def __call__(self, node_X, node_gamma, node_beta, output_grad):
        grad_beta = Op.__call__(self)
        grad_beta.inputs = [node_X, node_gamma, node_beta, output_grad]
        grad_beta.name = "batch_norm_grad_beta(%s, %s, %s)" % (node_X, node_gamma, node_beta)  
        self.eps = eps        
        return grad_beta

    def compute(self, node, input_vals):
        (X, gamma, beta, output_grad) = input_vals
        eps = self.eps        
        # Retrieving dimensions from X's shape
        mu = 1./N*np.sum(X, axis = 0)
        var = 1./N*np.sum((X-mu)**2, axis = 0)
        dbeta = np.sum(output_grad, axis=0)

        return dbeta

    def gradient(self, node, output_grad):
        return None 


class Layer(object):
    def forward(self, *input_nodes):
        raise NotImplementedError

class DenseLayer(Layer):

    def __init__(self, in_size, out_size, name='dense', initial='np.random.random_sample'):
        self.in_size = in_size
        self.out_size = out_size
        self.W = Variable(name='W')
        self.W.val_W = eval(initial)(W)
        self.name = dense

    def forward(self, node_X):
        self.inputs = [node_X]
        self.output = matmul(X, self.W)
        self.output.name = self.name
        self.output.op.compute


    def forward(self, input_node):
        output_node =  inv_op(mul_byconst_op(exp_op(mul_byconst_op(input_node, -1)),1))
        output_node.name = "sigmoid(%s)" % input_node.name
 




# Create global singletons of operators.
add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()

power_op = PowerOp()
cos_op = CosOp()
sin_op = SinOp()
tan_op = TanOp()
log_op = LogOp()
exp_op = ExpOp()
inv_op = InvOp()
sigmoid_op = SigmoidOp()

sign_op = SignOp()
relu_op = ReLuOp()

conv_op = ConvOp()
conv_grad_x_op = ConvGradXOp()
conv_grad_w_op = ConvGradWOp()

batch_norm_op = BatchNormOp()
batch_norm_grad_x_op = BatchNormGradXOp()
batch_norm_grad_gamma_op = BatchNormGradGammaOp()
batch_norm_grad_beta_op = BatchNormGradBetaOp()


class Executor:
    """Executor computes values for a given subset of nodes in a computation graph.""" 
    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list. 
        """
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all nodes.
        topo_order = find_topo_sort(self.eval_node_list)

        for node in topo_order:
            if node in node_to_val_map:
                continue
            else:
                input_vals = [node_to_val_map[input_note] for input_note in node.inputs]
                node_to_val_map[node] = node.op.compute(node, input_vals)
        """TODO: Your code here"""

        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results

def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    
    for cur_node in reverse_topo_order:

        output_grad = zeroslike_op(cur_node)
        for output_node_grad in node_to_output_grads_list[cur_node]:
            output_grad = add_op(output_grad, output_node_grad)
        node_to_output_grad[cur_node] = output_grad

        grad_wrt_inputs = cur_node.op.gradient(cur_node, output_grad)
        for input_node_idx in range(len(cur_node.inputs)):
            input_node = cur_node.inputs[input_node_idx]
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = [grad_wrt_inputs[input_node_idx]]
            else:
                node_to_output_grads_list[input_node].append(grad_wrt_inputs[input_node_idx])        
        
    """TODO: Your code here"""

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##############################
####### Helper Methods ####### 
##############################

def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    
    A simple algorithm is to do a post-order DFS traversal on the given nodes, 
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
