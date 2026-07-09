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
    loss = (-1)*round(loss/n, 4)

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

# Step 19 - conv2d_grad_weights (not yet solved)
# TODO: implement

# Step 20 - conv2d_grad_bias (not yet solved)
# TODO: implement

# Step 21 - conv2d_backward (not yet solved)
# TODO: implement

# Step 22 - maxpool2d_forward (not yet solved)
# TODO: implement

# Step 23 - scatter_grad_window (not yet solved)
# TODO: implement

# Step 24 - maxpool2d_backward (not yet solved)
# TODO: implement

# Step 25 - relu_forward (not yet solved)
# TODO: implement

# Step 26 - relu_backward (not yet solved)
# TODO: implement

# Step 27 - flatten_forward (not yet solved)
# TODO: implement

# Step 28 - flatten_backward (not yet solved)
# TODO: implement

# Step 29 - linear_forward (not yet solved)
# TODO: implement

# Step 30 - linear_grad_input (not yet solved)
# TODO: implement

# Step 31 - linear_grad_weights (not yet solved)
# TODO: implement

# Step 32 - linear_grad_bias (not yet solved)
# TODO: implement

# Step 33 - linear_backward (not yet solved)
# TODO: implement

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

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

