import numpy as np
    
from matplotlib import pyplot as plt


from stl10_input import DATA_PATH, read_all_images


from scipy.fftpack import dct, idct

#from colour import convert





images = linearize(read_all_images(DATA_PATH) / 255)
change_space(images, M0)

transformed = np.copy(images)
change_space(transformed, M1)
transformed **= (1/3)
change_space(transformed, M2)

oklab_dct = dct(dct(transformed, norm='ortho', axis=1), norm='ortho', axis=2)




def wlp(vector, p=2):
    f = np.abs(np.ndarray.flatten(vector))
    k = len(f) - np.argsort(f)
    return np.min(f * (k ** (1/p)))


ps = [0.05, 0.1, 0.5, 1, 1.5, 2]
def weaklps(transformed, ps, n=float('inf')):
    n = min(n, transformed.shape[0])
    res = np.zeros((len(ps), n))

    nrm = np.linalg.norm(
        transformed[:n,...].reshape((n,-1)),
        ord=1,
        axis=1)

    for i in range(n):
        for j in range(len(ps)):
            res[j,i] = wlp(transformed[i], ps[j]) / nrm[i]

    return res

averages = weaklps(oklab_dct, ps, 50)
plt.loglog();
plt.plot(ps, averages, 'o-');
plt.show()





        

    
