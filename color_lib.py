import numpy as np

# taken from http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
# and http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
# no idea if accurate...
def xyY_to_XYZ(point):
    x, y, Y = point
    return np.array([ x * Y / y, Y, (1 - x - y) * Y / y])

# https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
VARS = "defghijklmnopqrstuvwxyz"
def change_space(result_mat, transform_mat):
    var = VARS[:len(transform_mat.shape)]
    np.einsum(f"ab,...b->...a", transform_mat, result_mat, out=result_mat)

def linearize(V):
    return np.where(V <= +0.04045,
                    (V / 12.92),
                    ((V + 0.055) / 1.055)**2.4)

def applyGamma(V):
    return np.where(V <= +0.0031308,
                    V * 12.92,
                    1.055 * V**(1/2.4) - 0.055)


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
M1M0 = M1 @ M0
M1M0_inv = np.linalg.inv(M1 @ M0)
M2_inv = np.linalg.inv(M2)

def rgb_to_oklab(rgb_image):
    oklab_image = linearize(rgb_image)
    change_space(oklab_image, M1M0)
    oklab_image **= (1/3)
    change_space(oklab_image, M2)
    return oklab_image

def oklab_to_rgb(oklab_image):
    oklab_image = np.copy(oklab_image)
    change_space(oklab_image, M2_inv)
    oklab_image **= 3
    change_space(oklab_image, M1M0_inv)
    return applyGamma(oklab_image)
