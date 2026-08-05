"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    rows = len(matrix)
    columns = len(matrix[0])
    ans = []

    for r in range(0, rows):
        max_ele = float("-inf")
        for c in range(0, columns):
            if matrix[r][c] > max_ele:
                val = c 
                max_ele = matrix[r][c]

        ans.append(val)    

    return np.array(ans)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.

    rows = len(matrix)
    cols = len(matrix[0])
    ans = []

    for r in range(rows):
         max_val = float("-inf")
         for c in range(cols):
            if matrix[r][c] > max_val:
                max_val = matrix[r][c]
         ans.append([max_val])
    
    return np.array(ans)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    rows = len(matrix)
    cols = len(matrix[0])
    ans = []

    for r in range(rows):
        val = 0 
        for c in range(cols):
            val += matrix[r][c]
        ans.append([val])
    
    return np.array(ans)

# Step 4 - exp_shifted
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.

    rows = len(matrix)
    cols = len(matrix[0])
    ans = []

    for r in range(rows):
         max_val = float("-inf")
         for c in range(cols):
            if matrix[r][c] > max_val:
                max_val = matrix[r][c]
         ans.append([max_val])
    
    return np.array(ans)

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    n = len(logits)
    cols = len(logits[0])
    max_val = row_max(logits)

    ans = np.zeros_like(logits, dtype=float)
    for i in range(n):
        indi_max = max_val[i][0]
        for c in range(cols):
            ans[i][c] = np.exp(logits[i][c] - indi_max)
    
    return ans

# Step 5 - stable_softmax
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    shifted_exp = exp_shifted(logits)
    return shifted_exp/row_sum(shifted_exp)

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix

    n = len(labels)
    ans = [[] for _ in range(n)]

    for i in range(0, n):
        for j in range(0, num_classes):
            if j == labels[i]:
                ans[i].append(1)
            else:
                ans[i].append(0)
    return np.array(ans, dtype='float')

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.

    n = len(probs)
    m = len(probs[0])
    ans = []
    for i in range(0, n):
        for j in range(0, m):
            if j == labels[i]:
                ans.append(probs[i][j])
    
    return np.array(ans)

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    n = len(probs)
    true_class_probs = gather_true_class_probs(probs, labels)
    loss = 0
    for i in range(0, n):
        loss+=np.log(true_class_probs[i]+eps)
    loss = (-1)*(loss/n)

    return loss

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.

    n = len(labels)
    logit_vals = argmax_rows(logits_or_probs)
    acc = 0
    count = 0
    for i in range(0 ,n):
        if logit_vals[i] == labels[i]:
            count+=1
    
    return round(count/n, 4)

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2/fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.

    np.random.seed(seed)
    sigma = he_std(fan_in)
    return np.random.normal(0, sigma, shape).astype(np.float64)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length, dtype=np.float64)

# Step 13 - pad_2d
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.

    n,c,h,w = images.shape

    new_h = h + 2*pad 
    new_w = w + 2*pad 

    padded_image = np.zeros((n,c,new_h, new_w), dtype=images.dtype)

    for i in range(n):
        for j in range(c):
            for k in range(h):
                for l in range(w):
                    padded_image[i][j][k+pad][l+pad] = images[i][j][k][l]
    
    
    
    
    return padded_image

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    output_size = ((input_size-kernel + 2*padding)/stride) + 1
    return int(output_size)

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    # TODO: Unroll overlapping patches of a 4D image tensor into a 2D column matrix.
    padded_image = pad_2d(images, padding)
    n,c,h,w = images.shape
    out_h = output_spatial_size(h, kernel_h, stride, padding)
    out_w = output_spatial_size(w, kernel_w, stride, padding)

    output = np.zeros((n * out_h * out_w, c * kernel_h * kernel_w), dtype=images.dtype)

    row = 0 
    for i in range(n):
        for j in range(out_h):
            for k in range(out_w):
                patch = padded_image[i, :, j*stride:j*stride+kernel_h, k*stride:k*stride+kernel_w]
                output[row] = patch.flatten()
                row+=1 
    
    return output

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor
    n ,c, h, w = input_shape
    padded_h = h + 2*padding 
    padded_w = w + 2*padding 
    out_h = output_spatial_size(h, kernel_h, stride, padding)
    out_w = output_spatial_size(w, kernel_w, stride, padding)
    image = np.zeros((n,c,padded_h, padded_w))

    row = 0
    for i in range(n):
        for j in range(out_h):
            for k in range(out_w):
                patch = cols[row].reshape(c, kernel_h, kernel_w)
                image[i, :, j*stride:j*stride+kernel_h, k*stride:k*stride+kernel_w] += patch 
                row+=1

    if padding>0:
        return image[:, :, padding:-padding, padding:-padding]
    
    return image

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    # TODO: convolve x with weights using im2col, add bias, return output and a backprop cache.

    C_out, C_in, kernel_h, kernel_w = weights.shape
    X_reshaped = im2col(x, kernel_h, kernel_w, stride, padding)
    w_flat = weights.reshape(C_out, -1)
    Y = X_reshaped @ w_flat.T + bias 
    N, _, H, W_in = x.shape 
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W_in, kernel_w, stride, padding)

    out = Y.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
    cache = {
    'x_shape': x.shape,
    'weights': weights,
    'cols': X_reshaped,
    'stride': stride,
    'padding': padding,
    'kernel_h': kernel_h,
    'kernel_w': kernel_w
    }

    return out, cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    # TODO: backprop d_out through the conv input using col2im

    N, C_out, out_h ,out_w = d_out.shape 
    d_Y = d_out.transpose(0, 2, 3, 1).reshape(N*out_h*out_w, -1)
    weights = cache['weights']
    W_flat = weights.reshape(C_out, -1)
    d_X_cols = d_Y @ W_flat 
    x_shape, kernel_h, kernel_w, stride, padding = cache['x_shape'], cache['kernel_h'], cache['kernel_w'], cache['stride'], cache['padding']
    d_X = col2im(d_X_cols, x_shape, kernel_h, kernel_w, stride ,padding)
    return d_X

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.
    N, C_out, out_h ,out_w = d_out.shape 
    d_Y = d_out.transpose(0, 2, 3, 1).reshape(N*out_h*out_w, -1)
    cols, kernel_h ,kernel_w = cache['cols'], cache['kernel_h'], cache['kernel_w']
    C_in = cache['weights'].shape[1]  
    d_W_flat = cols.T @ d_Y 
    d_W_flat = d_W_flat.T.reshape(C_out, C_in, kernel_h, kernel_w)

    return np.array(d_W_flat)

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    return d_out.sum(axis=(0,2,3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache
    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)
    return dx, dW, db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    # TODO: run 2D max pooling and cache the in-window argmax of each output cell.
    n,c,h,w = x.shape 
    out_h = output_spatial_size(h, kernel, stride, 0)
    out_w = output_spatial_size(w, kernel, stride, 0)

    out = np.zeros((n, c, out_h, out_w), dtype=np.float64)
    argmax = np.zeros((n, c, out_h, out_w), dtype=np.int64)

    for i in range(n):
        for j in range(c):
            for k in range(out_h):
                for l in range(out_w):
                    patch = x[i, j, k*stride:k*stride+kernel,
                                    l*stride:l*stride+kernel]
                    out[i, j, k, l] = patch.max()
                    argmax[i,j,k,l] = np.argmax(patch)
    
    cache = {
        'x_shape' : x.shape, 
        'argmax' : argmax, 
        'kernel' : kernel, 
        'stride' : stride
    }

    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    # TODO: place grad_value at the argmax position within a (kernel, kernel) zero array.
    window = np.zeros((kernel, kernel), dtype=np.float64)
    row = argmax_index // kernel 
    col = argmax_index % kernel   
    window[row][col] = grad_value 
    return window

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    # TODO: scatter each d_out value to the cached argmax position in its window
    
    n,c,out_h,out_w = d_out.shape 
    argmax  = cache['argmax']
    x_shape = cache['x_shape']
    kernel = cache['kernel']
    stride = cache['stride']
    d_x = np.zeros(x_shape, dtype=np.float64)

    for i in range(n): 
        for j in range(c):
            for k in range(out_h):
                for l in range(out_w):
                    grad_val = d_out[i,j,k,l]
                    idx = argmax[i,j,k,l]
                    window = scatter_grad_window(grad_val, idx, kernel)
                    d_x[i, j, k*stride:k*stride+kernel, l*stride:l*stride+kernel] += window 
    
    return np.array(d_x)

# Step 25 - relu_forward
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.
    
    out = np.maximum(0, x)
    cache = {
        'x':x
    }

    return out, cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    x = cache['x']
    d_relu = d_out * (x>0)
    return d_relu

# Step 27 - flatten_forward
def flatten_forward(x):
    # TODO: reshape a 4D feature map into a 2D batch matrix and cache the original shape
    n,c,h,w = x.shape
    out = x.reshape(n, c*h*w)
    cache = {
        'x_shape' : x.shape
    }
    return out, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    n, c, h, w = cache['x_shape']
    dx = d_out.reshape(n,c,h,w)
    return dx

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    # TODO: compute X @ W + b and cache the inputs needed for backprop.

    x_out = x @ weights + bias 
    cache = {
        'x' : x, 
        'weights' : weights
    }

    return x_out, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    w = cache['weights']
    d_x = d_out @ w.T 
    return d_x

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
    d_w = x.T @ dout
    return d_w

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    d_b = dout.sum(axis=0)
    return d_b

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache
    x = cache['x']
    dx = linear_grad_input(dout, cache)
    dW = linear_grad_weights(x, dout)
    db = linear_grad_bias(dout)

    return dx, dW, db

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    # TODO: return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).
    m = logits.max(axis=1, keepdims=True)
    shifted = logits-m 
    exp = np.exp(shifted)
    s = exp.sum(axis=1, keepdims=True)
    p = exp/s 
    N = logits.shape[0]
    true_class_probs = p[np.arange(N), y]
    losses = -np.log(true_class_probs)
    return float(losses.mean())

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    # TODO: return the fused softmax-cross-entropy gradient of shape (N, C).
    N = logits.shape[0]
    p = stable_softmax(logits)
    dlogits = (p-one_hot(y, logits.shape[1])) / N 
    return dlogits + 0.0

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).
    param = param - lr*grad
    return param

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.
    m_t = beta_one*m + (1-beta_one)*grad 
    return m_t

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    v_t = beta_two*v + (1-beta_two) * (grad)**2 
    return v_t

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    return moment/(1-beta**t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    param = param - lr*(m_hat/(np.sqrt(v_hat) + eps))
    return param

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    # TODO: chain the four Adam helpers and return (new_param, new_m, new_v)
    m_t = adam_update_m(m, grad, beta_one)
    m_v = adam_update_v(v, grad, beta_two)
    new_m = adam_bias_correct(m_t, beta_one, t)
    new_v = adam_bias_correct(m_v, beta_two, t)
    new_param = adam_param_step(param, new_m, new_v, lr, eps)
    return new_param, m_t, m_v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    # TODO: Build He-initialized weights and a zero bias for a single conv layer.
    fan_in = in_channels * kernel_size * kernel_size
    W = he_init((out_channels, in_channels, kernel_size, kernel_size), fan_in, seed)
    b = init_zero_bias(out_channels)
    
    conv_dict ={
        'W' : W, 
        'b' : b
    }

    return conv_dict

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}
    fan_in = in_features 
    W = he_init((in_features,out_features), fan_in, seed)
    b = init_zero_bias(out_features)

    lin_dict = {
        'W': W , 
        'b': b
    }
    
    return lin_dict

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    # TODO: build conv1, conv2, fc1, fc2 with the right shapes and return them in a dict.
    
    conv1 = init_conv_layer(6, in_channels ,5, seed)
    conv2 = init_conv_layer(16, 6, 5, seed+1)
    fc1 = init_linear_layer(256,  120)
    fc2 = init_linear_layer(120, num_classes)

    params = {
        'conv1':conv1, 
        'conv2':conv2,
        'fc1':fc1, 
        'fc2':fc2
    }

    return params

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # TODO: run conv2d -> relu -> maxpool2d and return (out, cache_dict)
    out_conv2d, cache_conv2d = conv2d_forward(x, W, b, stride, pad)
    out_relu, cache_relu = relu_forward(out_conv2d)
    out, cache = maxpool2d_forward(out_relu, pool_size, pool_size)

    cache = {
        'conv_cache' : cache_conv2d, 
        'relu_cache' : cache_relu, 
        'pool_cache' : cache,
    }

    return out, cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: run flatten -> linear -> relu -> linear and return logits plus a cache dict.
    x_flat, flatten_cache = flatten_forward(x)
    w1, b1 = fc1['W'], fc1['b']
    w2, b2 = fc2['W'], fc2['b']

    x_lin_1, cache_lin_1 = linear_forward(x_flat, w1,b1)
    x_relu_forward, cache_relu = relu_forward(x_lin_1)
    x_lin_2, cache_lin_2 = linear_forward(x_relu_forward, w2, b2)

    cache = {
        'flatten_cache' : flatten_cache, 
        'fc1_cache' : cache_lin_1,
        'relu_cache' : cache_relu, 
        'fc2_cache' : cache_lin_2
    }

    return x_lin_2, cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).
    #Given to us 
    pool_size = 2 
    stride = 1
    padding = 0 

    conv1, conv2, fc1, fc2  = params['conv1'], params['conv2'], params['fc1'], params['fc2']
    
    W1_conv, b1_conv = conv1['W'], conv1['b']

    conv1_out, cache_conv1 = forward_conv_block(x, W1_conv, b1_conv, pool_size, stride, padding)

    W2_conv, b2_conv = conv2['W'], conv2['b']
    
    conv2_out, cache_conv2 = forward_conv_block(conv1_out, W2_conv, b2_conv, pool_size, stride, padding)

    out, cache_forward_classifier = forward_classifier_block(conv2_out, fc1, fc2)

    cache = {
        'block1':cache_conv1, 
        'block2':cache_conv2, 
        'classifier':cache_forward_classifier
    }

    return out, cache

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # TODO: backprop dout through the cached pool, relu, and conv layers in reverse order.
    
    d_pool = maxpool2d_backward(dout , cache['pool_cache'])
    d_relu = relu_backward(d_pool, cache['relu_cache'])
    dx, dW, db = conv2d_backward(d_relu, cache['conv_cache'])

    return (dx, dW, db)

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # TODO: backprop through fc2 -> relu -> fc1 -> flatten using the cached values

    dfc_2, dW2, db2 = linear_backward(dlogits, cache['fc2_cache'])
    d_relu = relu_backward(dfc_2, cache['relu_cache'])
    dfc_1, dW1, db1 = linear_backward(d_relu, cache['fc1_cache'])

    dx = flatten_backward(dfc_1, cache['flatten_cache'])

    cache = {
        'dx':dx,
        'fc1': {'dW':dW1, 'db':db1},
        'fc2': {'dW':dW2, 'db':db2}

    }

    return cache

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    # TODO: walk classifier and conv block caches in reverse to assemble all gradients

    clf_grads = backward_classifier_block(dlogits, caches['classifier'])
    df2, dW2,db2 = backward_conv_block(clf_grads['dx'], caches['block2'])
    df1, dW1, db1 = backward_conv_block(df2, caches['block1'])

    return {
    'conv1': {'dW': dW1, 'db': db1},
    'conv2': {'dW': dW2, 'db': db2},
    'fc1': clf_grads['fc1'],
    'fc2': clf_grads['fc2'],
}

# Step 51 - lenet_predict
def lenet_predict(x, params):
    # TODO: Return the argmax class index per sample from a LeNet forward pass.
    logits, cache = lenet_forward(x, params)
    return np.argmax(logits, axis=1)

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    # TODO: Return (x, y) for a reproducible synthetic NCHW image dataset.
    rng = np.random.default_rng(seed)
    y = rng.integers(num_classes, size=num_samples)
    x = rng.standard_normal((num_samples, in_channels, image_size, image_size))
    x += (y-(num_classes-1)/2).reshape(-1,1,1,1)
    return x,y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    # TODO: return a reproducible permutation of [0, n) as an int ndarray of shape (n,).
    rng = np.random.RandomState(seed)
    return rng.permutation(n)

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    # TODO: partition x and y into train and test halves using a shared shuffled order.
    N = x.shape[0]
    idx = shuffle_indices(N, seed)
    n_test = int(N*test_fraction)
    test, train = idx[:n_test] , idx[n_test:]
    return x[train], y[train], x[test], y[test]

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

