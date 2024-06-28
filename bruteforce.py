import numpy as np
import os
from stl10_input import DATA_PATH, read_all_images
from color_lib import rgb_to_oklab, oklab_to_rgb
from matplotlib import pyplot as plt
from scipy import sparse

images = rgb_to_oklab(read_all_images(DATA_PATH) / 255)

def bw(image, i=0):
    im2 = np.zeros(image.shape + (3,))
    im2[:,:,i] = image
    if i != 0:
        im2[:,:,0] = 0.5
    return oklab_to_rgb(im2)
plt.subplot(2,2,1).imshow(bw(images[0,:,:,0]))
plt.subplot(2,2,2).imshow(bw(images[0,:,:,1], 1))
plt.subplot(2,2,3).imshow(bw(images[0,:,:,2], 2))
plt.subplot(2,2,4).imshow(oklab_to_rgb(images[0]))
plt.show()


from scipy.fftpack import dct, idct




rng = np.random.default_rng(77)
DIM = 96


def idct2d(square, axes=(-2,-1)):
    step1 = idct(square, norm='ortho', axis=axes[0])
    step2 = idct(square, norm='ortho', overwrite_x=True, axis=axes[0])
    return step2

def get_ax_slice(axis, x, *s):
    s = (axis-1)*(slice(None),) \
               + (slice(*s),)\
               + (len(x.shape) - axis) * (slice(None),)

class blind_slicer:
    def __init__(self, n_axes, axes):

        try:
            self.axes = tuple(((a + n_axes if a < 0 else a) for a in axes))
        except TypeError:
            self.axes = (axes + n_axes if axes < 0 else axes),
        
        self.n_axes = n_axes

    def __getitem__(self, s):
        if len(self.axes) == 1:
            if not isinstance(s, tuple):
                s = (s,)
            
        assert len(s) == len(self.axes)

        r = [slice(None)] * self.n_axes
        for i, ix in zip(self.axes, s):
            r[i] = ix

        return tuple(r)
        
    
class axis_slicer:
    def __init__(self, arr, axes):        
        self.slicer = blind_slicer(len(arr.shape), axes)
        self.arr = arr

    def __getitem__(self, s):
        return self.arr[self.slicer[s]]

    def __setitem__(self, s, r):
        self.arr[self.slicer[s]] = r

    def result(self):
        return self.arr

def idwt_step(x, axis=-1):
    assert x.shape[axis] % 2 == 0 and x.shape[axis]
    half = x.shape[axis] // 2

    
    ix = axis_slicer(np.zeros_like(x), axis)



    ix[0::2] = x[ix.slicer[:half]] + x[ix.slicer[half:]]
    ix[1::2] = x[ix.slicer[:half]] - x[ix.slicer[half:]]
    out = ix.result()
    out *= np.sqrt(0.5)

    return out

def idwtn(x, n, axis=-1, overwrite_x = False):
    if not overwrite_x:
        x = np.copy(x)
    l = x.shape[axis]
    x = axis_slicer(x, axis)
    for i in range(n-1, -1, -1):
        x[:(l >> i)] = idwt_step(x[:(l >> i)], axis)
    return x.result()

def idwt2d(x, n, axes=(-2,-1), scramble=False):
    x = np.copy(x)
    lx = x.shape[axes[0]]
    ly = x.shape[axes[1]]
    
    ix = axis_slicer(x, axes)
    
    for i in range(n, -1, -1):
        nx = lx >> i
        ny = ly >> i
        if scramble:
            hx = nx >> 1
            hy = nx >> 1

            ix[hx:nx,   :ny] = idct(ix[hx:nx,   :ny], axis=axes[0], overwrite_x=True, norm='ortho')
            ix[  :hx, hy:ny] = idct(ix[  :hx, hy:ny], axis=axes[0], overwrite_x=True, norm='ortho')
            ix[  :nx, hy:ny] = idct(ix[  :nx, hy:ny], axis=axes[1], overwrite_x=True, norm='ortho')
            ix[hx:nx,   :hy] = idct(ix[hx:nx,   :hy], axis=axes[1], overwrite_x=True, norm='ortho')
            
        
        
        ix[:nx,:ny] = idwt_step(ix[:nx, :ny], axes[0])
        ix[:nx,:ny] = idwt_step(ix[:nx, :ny], axes[1])

    return ix.result()

    

class FastJLTransformer:
    def __init__(self, rand, d, n):
        self.length = d
        self.max_idx = 1 << int(np.ceil(np.log2(d)))
        self.indices = np.array(sorted(rand.choice(range(d), n, replace=False)))
        self.multiplier = rand.choice([-1,1], d, replace=True)

    def __call__(self, x, axis=-1):
        assert self.length == x.shape[axis]
        if axis < 0:
            axis += len(x.shape)
        return axis_slicer(
        
            # DCT is almost as good as 
            
            #dct(
                x * self.multiplier.reshape((1,)*axis + (-1,) + (1,)*(len(x.shape)-axis-1))
            #, axis=axis, overwrite_x=True)

            , axis)[self.indices]
        # if axis < 0:
            # axis += len(x.shape)


        # shape = list(x.shape)
        # shape[axis] = len(self.indices)
        # output = np.zeros(shape)
        
        # temp = np.zeros(self.max_idx)
        # shape[axis] = None
        
        # for idx in self._row_enumerate(shape, axis):
            # output[idx] = self._fwht(temp, x[idx])
        
        # return output

    def _row_enumerate(self, shape, axis, so_far = ()):
        if len(so_far) == axis:
            yield from self._row_enumerate(shape, axis, so_far + (slice(None),))
        elif len(so_far) == len(shape):
            yield so_far
        else:
            for i in range(shape[len(so_far)]):
                yield from self._row_enumerate(shape, axis, so_far + (i,))
    

    def _fwht(self, q, n):
        q = np.reshape(q, (-1, 1))
        q[0:len(n), :] = n[:,np.newaxis]
        q[len(n):, :] = 0
        while q.shape[0] > 1:
            q[0::2], q[1::2] = q[0::2] + q[1::2], q[0::2] - q[1::2]
            q = np.reshape(q, (q.shape[0] >> 1, q.shape[1] << 1))

        return np.ravel(q)[self.indices]

def normalized(a, order=2, axis=-1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)




fat = np.eye(96*96).reshape((-1, 96, 96))
def transform_fn(x, axes):
    #return idct2d(x, axes)
    return idwt2d(x, 4, axes)




fat = transform_fn(fat, (1,2)).reshape((-1, 96*96))
fat2 = np.zeros((96*96*2,96*96,2))
fat2[:96*96,:,0] = fat
fat2[96*96:,:,1] = fat
fat2 = fat2.reshape(-1, 96*96*2)





transformer1 = FastJLTransformer(rng, 96*96, int(np.ceil(96*96*(3/4))))
transformer2 = FastJLTransformer(rng, 96*96*2, int(np.ceil(96*96*2*(3/16))))


basis1 = transformer1(fat)
# for i in range(0, 64):
    # plt.subplot(16,16,i+1).imshow(basis1[:, i].reshape(96,96))
# plt.show()

basis2 = transformer2(fat2)
inputs1 = transformer1(images[:,:,:,0].reshape((-1, 96*96)), axis=1)
inputs2 = transformer2(images[:,:,:,1:].reshape((-1, 96*96*2)), axis=1)

def pursuit(row_basis, vector, orthonormalize=False):
    row_basis = row_basis.copy()
    y = np.linalg.norm(row_basis, axis=1)
    y[y==0] = 1
    row_basis /= y[:,np.newaxis]
    
    

    starting = np.linalg.norm(vector, ord=1)
    goal = starting * (0.125 if orthonormalize else 1e-2)
    print(starting, "goal =", goal)
    
    result = np.zeros(row_basis.shape[0])
##    orthos = sparse.eye(row_basis.shape[0], format='csr')
##    orthos *= sparse.dia(1/y, format='csr')
    orthonormalized_to = 0
    
    for i in range(row_basis.shape[0]-1,0,-1):
        v = row_basis @ vector
        idx = np.argmax(np.abs(v))
        m = v[idx]
        vector -= m * row_basis[idx]
        result[idx] += m / y[idx]

##        if i != idx:
##            row_basis[i,:], row_basis[idx,:] = row_basis[idx,:], row_basis[i,:]
##            orthos[i,:], orthos[idx,:] = orthos[idx,:], orthos[i,:]
##            v[idx],v[i] = v[i],v[idx]

        # orthonormalize
        if orthonormalize:
            v = row_basis[:i] @ row_basis[i]
            row_basis -= v[:-1,np.newaxis].T @ row_basis[i, np.newaxis]
            orthos -= v[:-1,np.newaxis].T @ orthos[i, np.newaxis]
            l2 = np.linalg.norm(row_basis, axis=1)
            l2[l2 == 0] = 1
            row_basis /= l2[:,np.newaxis]
            orthos /= l2[:,np.newaxis]
            
        

        if not i & 0x7F:
            r = np.linalg.norm(vector, ord=1)
            print(r)
            if r <= goal:
                break
            

    return result, vector

def channel(basis, vector):
    result, remains = pursuit(basis, vector, orthonormalize=False)
    result = result.reshape((96,96))
    result = transform_fn(result, (0,1))
    return result

image = np.zeros((96, 96, 3))
for i in range(min(10, images.shape[0])):
    r1, _ = pursuit(basis1, inputs1[i], orthonormalize=False)
    image[:,:,0] = transform_fn(r1.reshape((96,96)), (0,1))
    image[:,:,1:] = 0

    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    
    ax1.imshow(oklab_to_rgb(images[i]))
    ax2.imshow(oklab_to_rgb(image))
    
    
    r2, _ = pursuit(basis2, inputs2[i])
    image[:,:,1:] = transform_fn(r2.reshape((96,96,2)), (0,1))

    ax3.imshow(oklab_to_rgb(image))
    
    plt.show()





