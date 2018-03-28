import autodiff
from autodiff_test import *
import numpy as np

X = [[1,2,5],[3,4,5],[6,7,8],[3,4,7],[4,3,2],[3,3,1]]
y = [1,0,0,1,1,0]

X = np.array(X)
y = np.array(y)

batch_size = 10
input_size = 5
output_size = 1

h1_size = 8

X = np.random.random_sample([batch_size,input_size])
y = np.random.random_sample([batch_size,output_size])

W1_init = np.random.random_sample([input_size,h1_size])
W2_init = np.random.random_sample([h1_size,output_size])

def two_layers_mlp(input_vals):
    W1_init,W2_init = input_vals[:input_size*h1_size], input_vals[input_size*h1_size:]
    W1_init = np.array(W1_init).reshape([input_size,h1_size])
    W2_init = W2_init.reshape([h1_size,output_size])
    x2 = np.matmul(X, W1_init)
    x3 = np.matmul(x2, W2_init)
    return np.sum(x3)
    # err = np.sum((y-x3) * (y-x3) * (1/batch_size))
    
def param_list_to_vec(input_list):
    param_list = []
    for param in input_list:
        param_list.append(param.reshape(-1))
    return np.concatenate(param_list,axis=0)

num_input_vals = param_list_to_vec([W1_init,W2_init])

def num_grad(fun, input_vals, eps=1e-3):
    y0 = eval(fun)(input_vals)
    n = len(input_vals)
    grad = []
    for i in range(n):
        new_input_vals = input_vals.copy()
        new_input_vals[i] = input_vals[i] + eps
        y1 = eval(fun)(new_input_vals)
        grad.append((y1-y0) / eps)
    return np.array(grad)

def two_layers_mlp_ad():
    x1 = ad.Variable(name='x1')
    # y = ad.Variable(name='y')
    w1 = ad.Variable(name='w1')
    w2 = ad.Variable(name='w2')
    x2 = x1.matmul(w1)
    x3 = x2.matmul(w2)
    loss = x3
    # err = x2-y
    # loss = (err * err) * (1/batch_size)

    grad_w1,grad_w2 = ad.gradients(loss,[w1,w2])

    excutor = ad.Executor([loss,grad_w1,grad_w2])
    
    y_val, grad_w1_val, grad_w2_val = excutor.run(feed_dict={x1:X,w1:W1_init,w2:W2_init})

    return y_val, grad_w1_val, grad_w2_val

def test_two_layers():
    y_val_ad, grad_w1_val_ad, grad_w2_val_ad = two_layers_mlp_ad()

    y_val = two_layers_mlp(num_input_vals)

    grads_val = num_grad('two_layers_mlp',num_input_vals)
    grad_w1_val, grad_w2_val = grads_val[:input_size*h1_size],grads_val[input_size*h1_size:]
    grad_w1_val = grad_w1_val.reshape([input_size,h1_size])
    grad_w2_val = grad_w2_val.reshape([h1_size,output_size])

test_two_layers()

""" The two results agree!! """
