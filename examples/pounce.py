# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from sklearn.decomposition import PCA
import jax.numpy as jnp

from coreax.kernel import rbf_kernel, median_heuristic, stein_kernel_pc_imq_element, rbf_grad_log_f_X
from coreax.kernel_herding import stein_kernel_herding_block


def main(dir_="./examples/data/pounce"):

    # path to directory containing video as sequence of images
    fn = "pounce.gif"
    os.makedirs(f"{dir_}/coreset", exist_ok=True)

    # read in as video. Frame 0 is missing A from RGBA.
    Y_ = np.array(imageio.v2.mimread(f"{dir_}/{fn}")[1:])
    Y = Y_.reshape(Y_.shape[0], -1)

    # run PCA to reduce the dimension of the images whilst minimising effects on some of the statistical
    # properties, i.e. variance.
    p = 25
    pca = PCA(p)
    X = pca.fit_transform(Y)

    # request a 10 frame summary of the video
    C = 10

    # set the bandwidth parameter of the underlying RBF kernel
    N = min(X.shape[0], 1000)
    idx = np.random.choice(X.shape[0], N, replace=False)
    nu = median_heuristic(X[idx])

    # define the kernel
    k = lambda x, y : rbf_kernel(x, y, jnp.float32(nu)**2)/(nu * jnp.sqrt(2. * jnp.pi))
    weighted = True

    # run Stein kernel herding in block mode to avoid GPU memory issues
    coreset, Kc, Kbar = \
        stein_kernel_herding_block(X, C, stein_kernel_pc_imq_element, rbf_grad_log_f_X, nu=nu, max_size=1000)

    # sort the coreset ready for producing the output video
    coreset = jnp.sort(coreset)
    print('Coreset:', coreset)

    # Save a new video. Y_ is the original sequence with dimensions preserved
    coreset_images = Y_[coreset]
    imageio.mimsave(f"{dir_}/coreset/coreset.gif", coreset_images)

    # plot to visualise which frames were chosen from the sequence
    # action frames are where the "pounce" occurs
    action_frames = np.arange(63, 85)
    x = np.arange(N)
    y = np.zeros(N)
    y[coreset] = 1.
    z = np.zeros(N)
    z[jnp.intersect1d(coreset, action_frames)] = 1.
    plt.figure(figsize=(20, 3))
    plt.bar(x, y, alpha=.5)
    plt.bar(x, z)
    plt.xlabel("Frame")
    plt.ylabel("Chosen")
    plt.tight_layout()
    plt.savefig(f"{dir_}/coreset/frames.png")
    plt.close()


if __name__ == '__main__':
    main()
