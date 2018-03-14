# hw6
import numpy as np
import autodiff as ad
from math import *

# def num_grad(fun, input_vals, eps=1e-3):
#     y0 = eval(fun)(input_vals)
#     n,m = input_vals.shape
#     grad = np.zeros([n,m])
#     for i in range(n):
#         for j in range(m):
#             eps_mat = np.zeros([n,m])
#             eps_mat[i,j] = eps
#             y1 = eval(fun)(input_vals+eps_mat)
#             grad[i,j] = (y1-y0) / eps
#     return grad



def num_grad(fun, input_vals, eps=1e-3):
    y0 = eval(fun)(input_vals)
    n = len(input_vals)
    grad = []
    for i in range(n):
        new_input_vals = list(input_vals)
        new_input_vals[i] = input_vals[i] + eps
        y1 = eval(fun)(new_input_vals)
        grad.append((y1-y0) / eps)
    return grad



def ex6(x_lst):
    """X.shape = (3,1) """
    [x1,x2,x3] = x_lst
    return (sin(x1+1)+cos(2*x2))*tan(log(x3)) + (sin(x2+1)+cos(2*x1))*exp(1+sin(x3))


x_lst = [np.ones(1),2*np.ones(1),4*np.ones(1)]
y = ex6(x_lst)
[num_grad_x1_val, num_grad_x2_val, num_grad_x3_val] = num_grad('ex6', x_lst, eps=1e-3)


def display(name, real_val, sym_val):
    print("The real value for %s is %3.6f. The AD value is %3.6f. Their difference is %3.6f.\n" 
            % (name, real_val, sym_val,abs(real_val-sym_val)))

def ex6_sym():
    x1 = ad.Variable(name='x1')
    x2 = ad.Variable(name='x2')
    x3 = ad.Variable(name='x3')

    lhs = ((x1+1).sin() + (2*x2).cos()) * (x3.log()).tan()
    rhs = ((x2+1).sin() + (2*x1).cos()) * (1+x3.sin()).exp()
    y = lhs + rhs

    grad_x1, grad_x2, grad_x3 = ad.gradients(y, [x1,x2,x3])

    executor = ad.Executor([y, grad_x1, grad_x2, grad_x3])

    x1_val, x2_val, x3_val = x_lst
    y_val, grad_x1_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x1:x1_val, x2:x2_val, x3:x3_val})

    return [y_val, grad_x1_val, grad_x2_val, grad_x3_val]

[ad_y_val, ad_grad_x1_val, ad_grad_x2_val, ad_grad_x3_val] = ex6_sym()


display('y', y, ad_y_val)
display('grad_x1',num_grad_x1_val, ad_grad_x1_val)
display('grad_x2',num_grad_x2_val, ad_grad_x2_val)
display('grad_x3',num_grad_x3_val, ad_grad_x3_val)
