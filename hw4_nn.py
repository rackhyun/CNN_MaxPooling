import numpy as np
from skimage.util.shape import view_as_windows


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
# batch_size = 8
# input_size = 32
# filter_width = 3
# filter_height = filter_width
# in_ch_size = 3
# num_filters = 8
#######

class nn_convolutional_layer:

    # W = 8*3*3*3
    # b = 1*8*1*1
    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        #######
        self.filter_width, self.filter_height, self.num_filters, self.in_ch_size = filter_width, filter_height, num_filters, in_ch_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    # input x.shape=(batch size, input channel size, in width, in height) (8,3,32,32)
    # out.shape = (batch size, num filter, out width, out height) (8,8,30,30)
    # out = N − F + 1 = 32 - 3 + 1 = 30

    #######
    def forward(self, x):
        # batch_size = x.shape[0]
        # in_ch_size = x.shape[1]
        y = view_as_windows(x, (x.shape[0], x.shape[1], self.filter_width, self.filter_height))
        # print('y shape : ', y.shape)

        # y.reshape (1, 1, w_size, h_size, batch_size, -1) // -1 mean in_ch_size * filter_width * filter_height
        y = y.reshape (y.shape[0], y.shape[1], y.shape[2],y.shape[3],y.shape[4],-1)

        # filter reshape w.reshape(num_filter, -1) // -1 mean in_ch_size * filter_width * filter_height
        ft = self.W.reshape(self.W.shape[0], -1)

        # return out as (1,1,w_size, h_size, batch_size, num_filter)
        out = y @ ft.T
        # print ('out shape : ', out.shape)

        # return out (1,1,w_size, h_size, batch_size, num_filter) -> (w_size, h_size, batch_size, num_filter)
        out = out.squeeze()
        # print('out shape after squeeze: ', out.shape)

        # for plus b -> b is just depend num_filter // 생각하기 쉽게 순서를 좀 바꿔주자
        # batch_size, num_filter_size, w_size, h_size
        out = np.transpose(out,(2,3,0,1))
        # print('out shape after transpose: ', out.shape)

        # apply bias
        out = out + self.b
        # print('out shape : ', out.shape)

        return out

    #######
    # Q2. Complete this method
    # x.shape = (batch size, input channel size, in width, in height) (8,3,32,32)
    # dLdy.shape = (batch size, num filter, out width, out height) (8,8,30,30)
    # dLdx.shape = (batch size, input channel size, in width, in height)
    # dLdW.shape = (num_filters, in_ch_size, filter_width, filter_height)
    # dLdb.shape = (1, num_filters, 1, 1)
    #######
    def backprop(self, x, dLdy):
        # calc dLdx
        # dLdy padding
        # print ('input dLdy : ', dLdy.shape, '\n')
        # set padding size, only axis out_width and out_height
        pad_size = self.filter_width - 1
        # pad only axis out_width and out_height with pad_size and value 0
        #       axis=0, axis=1, axis=2              , axis=3
        npad = ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size))
        dLdy_pad = np.pad(dLdy, npad, 'constant', constant_values=(0))
        # print ('dLdy_pad shape : ', dLdy_pad.shape)

        # view dLdy_pad as window (batch_size, num_filter, filter_width, filter_height)
        dLdy_view = view_as_windows(dLdy_pad, (dLdy_pad.shape[0], dLdy_pad.shape[1], self.filter_width, self.filter_height))
        # print('dLdy_view shape : ', dLdy_view.shape)

        # W reverse order > keep batch_size, in_ch_size
        w_rev = np.flip(self.W, (2,3))

        # flat dLdy_view and w_rev
        dLdy_view = dLdy_view.reshape(dLdy_view.shape[0], dLdy_view.shape[1], dLdy_view.shape[2], dLdy_view.shape[3], dLdy_view.shape[4], -1)
        # print('dLdy_view shape after reshape : ', dLdy_view.shape)
        # tranpose로 순서 변경 in_ch_size, num_filter, filter_width, filter_heigh
        w_rev = w_rev.transpose(1,0,2,3)
        # in_ch_size 외 계산이 쉽도록 flatten
        w_rev = w_rev.reshape(w_rev.shape[0], -1)
        # print ('w_rev shape after reshape: ', w_rev.shape)

        # convolution dLdy * w_reverse.T
        dLdx = dLdy_view @ w_rev.T
        dLdx = dLdx.squeeze()
        dLdx = dLdx.transpose(2,3,0,1)
        # print ('dLdx shape : ', dLdx.shape)

        #calc dLdw
        # x.shape = (batch size, input channel size, in width, in height) (8,3,32,32)
        # dLdy.shape = (batch size, num filter, out width, out height) (8,8,30,30)
        # dLdW.shape = (num_filters, in_ch_size, filter_width, filter_height)
        # convolution x * dLdy
        x_view = view_as_windows(x, (dLdy.shape[0],self.in_ch_size, dLdy.shape[2], dLdy.shape[3]))
        # print ('x_view shape : ', x_view.shape)
        # in_ch_size, view_out_width, view_out_height, batch_size, filter(dLdy)_width, filter(dLdy)_height
        x_view = x_view.transpose (0, 1, 5, 2, 3, 4, 6, 7)
        # print ('x_view shape after transpose: ', x_view.shape)
        #flat x_view keep in_ch_size, view_out_width, view_out_height
        x_view = x_view.reshape(x_view.shape[0], x_view.shape[1], x_view.shape[2], x_view.shape[3], x_view.shape[3], -1)
        # print('x_view shape after reshape: ', x_view.shape)
        # flat dLdy keep num filters
        dLdy_flat = dLdy.transpose(1,0,2,3)
        dLdy_flat = dLdy_flat.reshape(dLdy_flat.shape[0], -1)
        # print ('dLdy_flat shape : ', dLdy_flat.shape)
        dLdW = x_view @ dLdy_flat.T
        # print('dLdW shape :', dLdW.shape)
        dLdW = dLdW.squeeze()
        dLdW = dLdW.transpose (3, 0, 1, 2)
        # print('dLdW shape :', dLdW.shape)

        #calc dLdb
        # sum dLdy keep num_filter = axis1
        dLdb = np.sum(dLdy, (0,2,3))
        dLdb = dLdb.reshape(self.b.shape)
        # print ('dLdb shape : ', dLdb.shape)

        return dLdx, dLdW, dLdb


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######
        # self.max_index

    #######
    # Q3. Complete this method
    # x.shape=(batch size, input channel size, in width, in height)
    # out.shape = (batch size, input channel size, out width, out height)
    #######
    def forward(self, x):
        # print('[mp F]input x shape: ', x.shape)
        # print('[mp F]input x : ', x[0][0])
        # view as window
        # use numpy amax
        y = view_as_windows(x, (x.shape[0], x.shape[1], self.pool_size, self.pool_size), self.stride)
        # y.shape = (1, 1, 16, 16, 8, 3, 2, 2)
        # print ('[mp F]y shape: ', y.shape)
        y = y.squeeze()
        #[mp F]y shape after squeeze:  (16, 16, 8, 3, 2, 2)
        # print('[mp F]y shape after squeeze: ', y.shape)
        y = y.reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3], -1)
        # [mp F]y shape after reshape:  (16, 16, 8, 3, 4)
        # print('[mp F]y shape after reshape: ', y.shape)
        out = y.max(axis=4)
        # [mp F]out shape after max:  (16, 16, 8, 3)
        # print('[mp F]out shape after max: ', out.shape)
        self.max_index = y.argmax(axis=4)
        self.max_index = self.max_index.transpose(2, 3, 0, 1)
        # [mp F]max_index shape : (8, 3, 16, 16)
        # print('[mp F]max_index shape :', self.max_index.shape)
        # print('[mp F]max_index :', self.max_index[0][0])

        out = out.transpose(2, 3, 0, 1)
        # [mp F]out shape:  (8, 3, 16, 16)
        # print('[mp F]out shape after transpose: ', out.shape)
        return out

    #######
    # Q4. Complete this method
    # x.shape=(batch size, input channel size, in width, in height) (8,3,32,32)
    # dLdy.shape=(batch size, input channel size, out width, out height) (8,3,16,16)
    # return : dLdx.shape=(batch size, input channel size, in width, in height) (8,3,32,32)
    #######
    def backprop(self, x, dLdy):
        # print('[mp B]input x shape: ', x.shape)
        # print('[mp B]input dLdy shape: ', dLdy.shape)

        # max index값을 가지고 와서 해당 위치에 dLdy값을 넣어 줌

        tdx = np.zeros_like(x).astype(float)
        for i in np.arange(self.max_index.shape[0]):
            for j in np.arange(self.max_index.shape[1]):
                for k in np.arange(self.max_index.shape[2]):
                    for l in np.arange(self.max_index.shape[3]):
                        n = int(np.floor(self.max_index[i,j,k,l] / self.pool_size))
                        m = int (self.max_index[i,j,k,l] % self.pool_size)
                        #print ('n,m = (',n,',', m,')  ', self.max_index[i,j,k,l])
                        tdx[i,j, self.pool_size*k+n, self.pool_size*l+m] = dLdy[i,j,k,l]

        dLdx = tdx
        return dLdx

    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x) #(8, 3, 16, 16)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')