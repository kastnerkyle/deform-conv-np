import matplotlib
matplotlib.use("Agg")
from imageio import imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.signal as sg
import scipy as sp


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # from cs231n assignments
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    # from cs231n assignments
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    # from cs231n assignments
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# really, really useful reference
# http://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
def conv2(x, w, b, pad="same", stride=1, dilation=1, cut=True):
    if pad == "same":
        pad = max(w.shape[-1] // 2 - 1, 1)
    if dilation != 1:
        assert stride == 1
        assert cut
    n_filt, d_filt, h_filt, w_filt = w.shape

    N, C, H, W = x.shape
    h_stride_check = (H + 2 * pad - h_filt) % stride
    if h_stride_check != 0:
        if h_stride_check % 2 == 0:
            x = x[:, :, h_stride_check // 2:-h_stride_check // 2, :]
        elif h_stride_check // 2 >= 1:
            x = x[:, :, h_stride_check // 2:-h_stride_check // 2, :]
        elif h_stride_check // 2 == 0:
            x = x[:, :, 1:, :]
        else:
            raise ValueError("Check x")

    N, C, H, W = x.shape
    h_stride_check = (H + 2 * pad - h_filt) % stride
    assert h_stride_check == 0

    w_stride_check = (W + 2 * pad - w_filt) % stride
    if w_stride_check != 0:
        if w_stride_check % 2 == 0:
            x = x[:, :, :, w_stride_check // 2:-w_stride_check // 2 + 1]
        elif w_stride_check // 2 >= 1:
                x = x[:, :, :, w_stride_check // 2:-w_stride_check // 2]
        elif h_stride_check // 2 == 0:
                x = x[:, :, :, 1:]
        else:
            raise ValueError("Check y")

    N, C, H, W = x.shape
    w_stride_check = (W + 2 * pad - w_filt) % stride
    assert w_stride_check == 0

    if dilation != 1:
        h_dilation_check = H % dilation
        w_dilation_check = W % dilation
        if h_dilation_check != 0:
            if h_dilation_check // 2 >= 1:
                x = x[:, :, h_dilation_check // 2:-h_dilation_check // 2, :]
            else:
                x = x[:, :, 1:, :]
        if w_dilation_check != 0:
            if w_dilation_check // 2 >= 1:
                x = x[:, :, :, w_dilation_check // 2:-w_dilation_check // 2]
            elif w_dilation_check // 2 == 0:
                x = x[:, :, :, 1:]
        # space -> batch
        # NCHW
        N, C, H, W = x.shape
        assert H % dilation == 0
        assert W % dilation == 0
        # WCNH
        x = x.transpose(3, 1, 0, 2)
        new_N = dilation * N
        new_H = H // dilation
        x = x.reshape(W, C, new_N, new_H)
        # HCNW
        x = x.transpose(3, 1, 2, 0)
        new_W = W // dilation
        new_N = dilation * new_N
        x = x.reshape(new_H, C, new_N, new_W)
        # NCHW
        x = x.transpose(2, 1, 0, 3)

    n_x, d_x, h_x, w_x = x.shape
    h_out = (h_x - h_filt + 2 * pad) // stride + 1
    w_out = (w_x - w_filt + 2 * pad) // stride + 1

    assert h_out == int(h_out)
    assert w_out == int(w_out)

    h_out = int(h_out)
    w_out = int(w_out)

    x_col = im2col_indices(x, h_filt, w_filt, padding=pad, stride=stride)
    w_col = w.reshape(n_filt, -1)

    if b is None:
        out = np.dot(w_col, x_col)
    else:
        out = np.dot(w_col, x_col) + b[:, None]

    out = out.reshape(n_filt, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)
    if dilation != 1:
        #check the dims as being square
        # space -> batch
        # NCHW
        N, C, H, W = out.shape
        # HCNW
        out = out.transpose(2, 1, 0, 3)
        new_N = N // dilation
        new_W = W * dilation
        out = out.reshape(H, C, new_N, new_W)
        # WCNH
        out = out.transpose(3, 1, 2, 0)
        new_H = H * dilation
        new_N = new_N // dilation
        out = out.reshape(new_W, C, new_N, new_H)
        # NCHW
        out = out.transpose(2, 1, 3, 0)
    return out

def _to_bc_h_w_2(o):
    shp = o.shape
    o = o.transpose(2, 3, 0, 1)
    o = o.reshape((shp[2], shp[3], shp[0] * shp[1] // 2, 2))
    # bc h w 2
    return o.transpose(2, 0, 1, 3)

def _to_bc_h_w(x):
    shp = x.shape
    x = x.transpose(2, 3, 0, 1)
    x = x.reshape((shp[2], shp[3], shp[0] * shp[1]))
    return x.transpose(2, 0, 1)

def _to_b_c_h_w(x_o, shp):
    x_n = x_o.transpose(1, 2, 0)
    x_n = x_n.reshape((shp[2], shp[3], shp[0], shp[1]))
    return x_n.transpose(2, 3, 0, 1)


def conv_offset2(x, w, pad="same"):
    x_shape = x.shape
    o_offsets = conv2(x, w, None, pad="same")
    # clip these offsets?
    offsets = _to_bc_h_w_2(o_offsets)
    x_r = _to_bc_h_w(x)
    x_offset = np_batch_map_offsets(x_r, offsets)
    x_offset = _to_b_c_h_w(x_offset, x_shape)
    shp = o_offsets.shape
    o_offsets = o_offsets.transpose(0, 2, 3, 1).reshape((shp[0], shp[2], shp[3], shp[1] // 2, 2))
    o_offsets = o_offsets.transpose(0, 3, 1, 2, 4)
    return x_offset, o_offsets


def mid_crop(arr, crop_h, crop_w):
    n, c, h, w = arr.shape
    if h < crop_h:
        raise ValueError("Can't crop larger crop_h")
    if w < crop_w:
        raise ValueError("Can't crop larger crop_w")

    diff_h = abs(crop_h - h)
    diff_w = abs(crop_w - w)

    out = arr
    if diff_h == 0:
        out = out
    elif diff_h == 1:
        out = out[:, :, 1:, :]
    elif diff_h % 2 == 0:
        out = out[:, :, diff_h // 2:-diff_h // 2, :]
    else:
        out = out[:, :, diff_h // 2:-diff_h // 2, :]

    if diff_w == 0:
        out = out
    elif diff_w == 1:
        out = out[:, :, :, 1:]
    elif diff_w % 2 == 0:
        out = out[:, :, :, diff_w // 2:-diff_w // 2]
    else:
        out = out[:, :, :, diff_w // 2:-diff_w // 2]
    return out


def crop_match(*args):
    min_h = np.inf
    min_w = np.inf
    for arg in args:
        n, c, h, w = arg.shape
        if h < min_h:
            min_h = h
        if w < min_w:
            min_w = w

    crop_args = []
    for a in args:
        crop_args.append(mid_crop(a, min_h, min_w))

    return crop_args


def imshow(arr):
    plt.imshow(arr)
    plt.show()


def arrshow(arr, ax=None, cmap=None):
    # nchw -> hwc 
    i = arr[0].transpose(1, 2, 0)

    if cmap is None:
        cmap_n = "viridis"
    else:
        cmap_n = cmap
    if i.shape[-1] == 1:
        i = i[:, :, 0]
        if cmap is None:
            cmap_n = "gray"
        else:
            cmap_n = cmap

    if ax is None:
        plt.imshow(i, cmap=cmap_n)
        plt.show()
    else:
        ax.imshow(i, cmap=cmap_n)


def make_conv_params(input_dim, output_dim, kernel):
    #w_o = np.ones((output_dim, input_dim, kernel, kernel), dtype="float32")
    #b_o = np.ones((output_dim,), dtype="float32")
    random_state = np.random.RandomState(0)
    w_o = .01 * random_state.randn(output_dim, input_dim, kernel, kernel).astype("float32")
    b_o = np.zeros((output_dim,), dtype="float32")
    return w_o, b_o


# Modified from Felix Lau, MIT License
def np_map_coordinates(inp, coords, order=1):
    assert order == 1
    coords_lt = np.cast["int32"](np.floor(coords))
    coords_rb = np.cast["int32"](np.ceil(coords))
    coords_lb = np.asarray((coords_lt[:, 0], coords_rb[:, 1])).transpose(1, 0)
    coords_rt = np.asarray((coords_rb[:, 0], coords_lt[:, 1])).transpose(1, 0)

    def fancy_take(a1, ind):
        flat_ind = a1.shape[1] * ind[:, 0] + ind[:, 1]
        return np.take(inp, flat_ind).copy()

    vals_lt = fancy_take(inp, coords_lt)
    vals_rb = fancy_take(inp, coords_rb)
    vals_lb = fancy_take(inp, coords_lb)
    vals_rt = fancy_take(inp, coords_rt)

    coords_offset_lt = coords - np.cast["float32"](coords_lt)
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]
    return mapped_vals


def np_batch_map_coordinates(inp, coords, order=1):
    assert order == 1
    coords = coords.clip(0, inp.shape[1] - 1)
    mapped_vals = np.array([np_map_coordinates(inp, coord)
                            for inp, coord in zip(inp, coords)])
    return mapped_vals


def np_batch_map_offsets(inp, offsets):
    batch_size = inp.shape[0]
    input_size = inp.shape[1]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)
    mapped_vals = np_batch_map_coordinates(inp, coords)
    mapped_vals = mapped_vals.reshape(batch_size, input_size, input_size)
    return mapped_vals


def sp_map_coordinates(inp, coords, order=1):
    return sp.ndimage.interpolation.map_coordinates(inp, coords.T,
                                                    mode="nearest", order=order)

def sp_batch_map_coordinates(inp, coords, order=1):
    assert order == 1
    coords = coords.clip(0, inp.shape[1] - 1)
    mapped_vals = np.array([sp_map_coordinates(inp, coord)
                            for inp, coord in zip(inp, coords)])
    return mapped_vals


def sp_batch_map_offsets(inp, offsets):
    batch_size = inp.shape[0]
    input_size = inp.shape[1]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)
    mapped_vals = sp_batch_map_coordinates(inp, coords)
    mapped_vals = mapped_vals.reshape(batch_size, input_size, input_size)
    return mapped_vals


fname = "napoleon_sloth.png"
# rgba image
img_arr = imread(fname)
# rgb image
img_arr = img_arr[:200, :200, :3]

# gray image
img_arr = np.dot(img_arr, np.array([.2126, 0.7152, 0.0722]))

"""
plt.imshow(img_arr, cmap="gray")
plt.savefig("tmp.png")
"""

"""
inp = np.random.random((100, 100))
coords = np.random.random((200, 2)) * 99
r1 = sp_map_coordinates(inp, coords)
r2 = np_map_coordinates(inp, coords)
assert np.abs(r2 - r1).max() < 1E-6

inp = np.random.random((4, 100, 100))
coords = np.random.random((4, 200, 2)) * 99
rr1 = sp_batch_map_coordinates(inp, coords)
rr2 = np_batch_map_coordinates(inp, coords)
assert np.abs(rr2 - rr1).max() < 1E-6

inp = np.random.random((4, 100, 100))
offsets = np.random.random((4, 100, 100, 2)) * 2
rrr1 = sp_batch_map_offsets(inp, offsets)
rrr2 = np_batch_map_offsets(inp, offsets)
assert np.abs(rrr2 - rrr1).max() < 1E-6
"""


mb = img_arr[None, :, :, None]
# transpose to NCHW
mb = mb.transpose(0, 3, 1, 2)
minibatch_size = 1
in_dim = 1
n_filt = 32
kernel = 3

# conv parameters
np_w, np_b = make_conv_params(in_dim, n_filt, kernel)
a1_o = conv2(mb, np_w, np_b, pad="same")

# offset conv parameters
np_o_w2, _ = make_conv_params(n_filt, 2 * n_filt, kernel)
np_w2, np_b2 = make_conv_params(n_filt, n_filt, kernel)

a2_offset, a2_offset_p = conv_offset2(a1_o, np_o_w2, pad="same")
a2_o = conv2(a2_offset, np_w2, np_b2, pad="same")

# offset conv parameters
np_o_w3, _ = make_conv_params(n_filt, 2 * n_filt, kernel)
np_w3, np_b3 = make_conv_params(n_filt, n_filt, kernel)

a3_offset, a3_offset_p = conv_offset2(a2_o, np_o_w3, pad="same")
a3_o = conv2(a3_offset, np_w3, np_b3, pad="same")

# offset conv parameters
np_o_w4, _ = make_conv_params(n_filt, 2 * n_filt, kernel)
np_w4, np_b4 = make_conv_params(n_filt, n_filt, kernel)

# mb or a3_o?
a4_offset, a4_offset_p = conv_offset2(a3_o, np_o_w4, pad="same")
a4_o = conv2(a4_offset, np_w4, np_b4, pad="same")

"""
a1 = a1_o #conv2(mb, np_w, np_b, stride=1, dilation=1, pad="same")
a3 = conv2(a2_offset, np_w2, np_b2, stride=1, dilation=4, pad="same")
a5 = conv2(a3_offset, np_w3, np_b3, stride=1, dilation=8, pad="same")
a7 = conv2(a4_offset, np_w4, np_b4, stride=1, dilation=16, pad="same")
"""

"""
a1 = conv2(mb, np_w, np_b, stride=1, dilation=1, pad="same")
a3 = conv2(mb, np_w, np_b, stride=1, dilation=4, pad="same")
a5 = conv2(mb, np_w, np_b, stride=1, dilation=8, pad="same")
a7 = conv2(mb, np_w, np_b, stride=1, dilation=16, pad="same")
"""

a1 = a1_o
a3 = a2_o
a5 = a3_o
a7 = a4_o

a1, a3, a5, a7 = crop_match(a1, a3, a5, a7)

def stack(*args):
    return np.concatenate([a[..., None] for a in args], axis=-1)


def apply_weights(stacked_arr, hw, ww, sw):
    # stacked_arr is 5D
    # n_samples, n_channels, height, width, scales
    # hw height weights
    # ww width weights
    # sw scale weights
    a_w = ww[None] * hw[:, None]
    hww = a_w
    a_w = a_w[:, :, None] * sw[None, None]
    a_w = a_w[None, None]
    o = (a_w * stacked_arr).sum(axis=-1)
    return o, hww, a_w

r3 = stack(a1, a3, a5, a7)
#r3 = stack(a1, a3, a5)

random_state = np.random.RandomState(1999)
def h_x(size):
   hw = np.linspace(0, 1, size) - 0.5
   hw = -hw ** 2 + 0.5
   return hw

def w_x(size):
   ww = np.linspace(0, 1, size) - 0.5
   ww = -ww ** 2 + 0.5
   return ww

def s_x(size):
   sw = random_state.rand(size)
   return sw

hw = h_x(r3.shape[2])
ww = w_x(r3.shape[3])
sw = s_x(r3.shape[4])

r, hww, w = apply_weights(r3, hw, ww, sw)

def megaplot(im, final_im, stack, hw, ww, sw, kernel_offset):
    f = plt.figure()
    n_scales = stack.shape[-1]
    if n_scales < 3:
        raise ValueError("Cannot plot < 3 scales")
    n_y = n_scales + 3
    n_x = n_scales + 1
    gs1 = gridspec.GridSpec(n_y, n_x)
    a = []
    for i in range(n_scales + 1):
        a.append(plt.subplot(gs1[0, i]))
    ax2 = plt.subplot(gs1[1, 1:])
    ax3_2 = plt.subplot(gs1[2:n_x - 1, 1:])
    #ax3_1 = plt.subplot(gs1[2:n_x - 1, 0], sharey=ax3_2)
    ax3_1 = plt.subplot(gs1[2:n_x - 1, 0])
    ax4_1 = plt.subplot(gs1[n_x - 1, 0])
    #ax4_2 = plt.subplot(gs1[n_x - 1, 1:], sharex=ax3_2)
    ax4_2 = plt.subplot(gs1[n_x - 1, 1:])

    arrshow(im, a[0], cmap="gray")
    for i in range(1, n_scales + 1):
        sim = stack[0, kernel_offset:kernel_offset+1, :, :, i - 1][0]
        a[i].imshow(sim, cmap="gray")

    ax2.plot(sw)
    ax3_1.plot(hw, np.arange(len(hw)))
    ax3_1.invert_yaxis()
    ax4_1.imshow(hww, cmap="gray")
    ax4_2.plot(ww)
    arrshow(final_im[:, kernel_offset:kernel_offset+1], ax3_2)
    plt.show()

for j in range(n_filt):
    megaplot(mb, r, r3, hw, ww, sw, j)
    plt.savefig("tmp{}.png".format(j))
plt.show()
