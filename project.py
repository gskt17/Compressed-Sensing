
"""
S: desired domain to sample domain
F: sparse basis to desired domain

CS problem: find Fx to minimize ||x||_0 (or ||x||_1) given b
     where SFx = b


Noisy CS problem: find the x of maximum expectation given b where:
    where
        S(Fx + n) = b,
        n is a vector of zero-mean RVs.
            - n has possible correlation.
            - n has possible dependence on x.

    cov(n_i, n_j) = E[(n_i - E[n_i])(n_j - E[n_j])] = E[n_i * n_j]

    var((Sn)_i) = E[ ( (SAn)_i - E[(SAn)_i]) ** 2 ]
    = E[ (\sum (SF)_ij * n_j - E[\sum (SF)_ij * n_j]) ** 2]
    = E[ (\sum (SF)_ij * n_j - \sum (SF)_ij * E[n_j]) ** 2]
    = E[ (\sum (SF)_ij * E[n_j]) ** 2 ]    [E[n_j] = 0]
    = E[ (\sum (SF)_ij * n_j) * (\sum (SF)_ik * n_k) ]
    = E[ \sum_j \sum_k ((SF)_ij * n_j * (SF)_ik * n_k) ]
    = \sum^n_{j=1} \sum^n_{k=1} (SF)_ij * (SF)_ik * cov(n_j, n_k)

    = \sum^n_{j=1} ((SF)_ij)^2 * var(n_j) + 2 \sum^n_{j=2} \sum^{j-1}_{k=1} (SF)_ij * (SF)_ik * cov(n_j, n_k)


    We know that |cov(n_i, n_j)| <= sqrt(var(n_i) * var(n_j))

    So |(SF)_ij * (SF)_ik * cov(n_j,n_k)| <= |(SF)_ij * (SF)_ik| * sqrt(var(n_i) * var(n_j))
    
"""


import numpy as np
import tensorflow as tf
import os
import keras
import sys
from keras import layers, ops

from matplotlib import pyplot as plt


from stl10_input import DATA_PATH, read_all_images

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPUs:", tf.config.list_physical_devices('GPU'))



def linearize(V):
    return np.where(V <= +0.04045,
                    (V / 12.92),
                    ((V + 0.055) / 1.055)**2.4)
def applyGamma(V):
    return np.where(V <= +0.0031308,
                    V * 12.92,
                    1.055 * V**(1/2.4) - 0.055)  

images = linearize(read_all_images(DATA_PATH) / 255)

k = 20
rng = np.random.default_rng(77)
DIM = 96
n_samp = int(np.ceil(DIM * DIM * 0.05))
selections = rng.choice(range(DIM*DIM), size=n_samp, replace=False)
xsel = selections // DIM
ysel = selections %  DIM


inputs = np.reshape(images[:, xsel, ysel, :], (-1, n_samp*3))

model = keras.Sequential(
    [
        layers.Input((n_samp*3,)),
        layers.Dense(n_samp*5, activation='relu'),
        layers.Dense(n_samp*5, activation='relu'),
        layers.Dense(DIM*DIM*3, activation='relu'),
        layers.Reshape((DIM, DIM, 3)),
    ],
)



M0 = np.array([
    [+0.4124564, +0.3575761, +0.1804375],
    [+0.2126729, +0.7151522, +0.0721750],
    [+0.0193339, +0.1191920, +0.9503041]
])

M1 = np.array([
    [+0.8189330101, +0.0329845436, +0.0482003018],
    [+0.3618667424, +0.9293118715, +0.2643662691],
    [+0.1288597137, +0.0361456387, +0.6338517070],
])

M2 = np.array([
    [+0.2104542553, +1.9779984951, +0.0259040371],
    [+0.7936177850, -2.4285922050, +0.7827717662],
    [-0.0040720468, +0.4505937099, -0.8086757660],
])

def linear_sRGB_to_okLab(rgb):
    res = ops.einsum("ij,xyzj->xyzi", M1 @ M0, rgb)
    res = ops.power(ops.maximum(0, res), 1/3)
    res = ops.einsum("ij,xyzj->xyzi", M2, res)
    return res



def perceptual_loss(y_true, y_pred):
    return ops.sum(
        ops.abs(
            ops.subtract(
            linear_sRGB_to_okLab(y_true),
            linear_sRGB_to_okLab(y_pred))
        )
    )


model.compile(
    loss=perceptual_loss,
)



checkpoint_path = "./training_1/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)


if "-r" in sys.argv or True:
    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    model.fit(inputs, images, callbacks=[cp_callback])

else:
    
    
    model.load_weights(checkpoint_path)
##
##
##input("Prediction time.")
##
##
##

n = 10
predicted = model.predict(inputs[0:n])
scatter_sample = np.zeros_like(images[0, :, :, :])

for i in range(n):

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    ax1.imshow(images[i])
    scatter_sample[xsel, ysel, :] = images[i, xsel, ysel, :]
    ax2.imshow(scatter_sample)
    ax3.imshow(predicted[i])
    ax4.imshow(0.5 + (predicted[i] - scatter_sample) / 2)

    plt.savefig(f'img_{i}.png', bbox_inches='tight')
    plt.show()
