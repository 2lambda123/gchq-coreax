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

from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, random, vmap
from jax.typing import ArrayLike

from coreax.utils import KernelFunction


def k_mean_rand_approx(
    key: random.PRNGKeyArray,
    data: ArrayLike,
    kernel: KernelFunction,
    num_kernel_points: int = 1000,
    num_train_points: int = 2000,
) -> Array:
    r"""
    Approximate kernel row mean by regression on points selected randomly.

    Here, the kernel row mean is the matrix row sum divided by n.

    :param key: Key for random number generation
    :param data: The original :math:`n \times d` data
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    :return: Approximation of the kernel matrix row sum divided by n
    """
    data = jnp.asarray(data)
    num_data_points = len(data)
    k_pairwise = jit(
        vmap(vmap(kernel, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0)
    )

    # Randomly select points for kernel regression
    key, subkey = random.split(key)
    features_idx = random.choice(
        subkey, num_data_points, (num_kernel_points,), replace=False
    )
    features = k_pairwise(data, data[features_idx])

    # Select training points
    train_idx = random.choice(key, num_data_points, (num_train_points,), replace=False)
    target = k_pairwise(data[train_idx], data).sum(axis=1) / num_data_points

    # Solve regression problem.
    params, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)

    return features @ params


def k_mean_annchor_approx(
    key: random.PRNGKeyArray,
    data: ArrayLike,
    kernel: KernelFunction,
    num_kernel_points: int = 1000,
    num_train_points: int = 2000,
) -> Array:
    r"""
    Approximate kernel row mean by regression on points chosen by ANNchor construction.

    Here, the kernel row mean is the matrix row sum divided by n. The ANNchor
    implementation used can be found `here <https://github.com/gchq/annchor>`_.

    :param key: Key for random number generation
    :param data: The original :math:`n \times d` data
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    :return: Approximation of the kernel matrix row sum divided by n
    """
    data = jnp.asarray(data)
    n = len(data)
    kernel_pairwise = jit(
        vmap(vmap(kernel, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0)
    )
    # kernel_vector is a function R^d x R^d \to R^d
    kernel_vector = jit(vmap(kernel, in_axes=(0, None)))

    # Select point for kernel regression using ANNchor construction
    features = jnp.zeros((n, num_kernel_points))
    features = features.at[:, 0].set(kernel_vector(data, data[0]))
    body = partial(anchor_body, data=data, kernel_function=kernel_vector)
    features = lax.fori_loop(1, num_kernel_points, body, features)

    train_idx = random.choice(key, n, (num_train_points,), replace=False)
    target = kernel_pairwise(data[train_idx], data).sum(axis=1) / n

    # solve regression problem
    params, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)

    return features @ params


@partial(jit, static_argnames=["kernel_function"])
def anchor_body(
    idx: int,
    features: ArrayLike,
    data: ArrayLike,
    kernel_function: KernelFunction,
) -> Array:
    r"""
    Execute main loop of the ANNchor construction.

    :param idx: Loop counter
    :param features: Loop updateables
    :param data: Original :math:`n \times d` dataset
    :param kernel_function: Vectorised kernel function on pairs `(X,x)`: :math:`k:
                  \mathbb{R}^{n \times d} \times \mathbb{R}^d \rightarrow \mathbb{R}^n`
    :return: Updated loop variables `features`
    """
    features = jnp.asarray(features)
    data = jnp.asarray(data)

    max_entry = features.max(axis=1).argmin()
    features = features.at[:, idx].set(kernel_function(data, data[max_entry]))

    return features


def k_mean_nystrom_approx(
    key: random.PRNGKeyArray,
    data: ArrayLike,
    kernel: KernelFunction,
    num_points: int = 1000,
) -> Array:
    r"""
    Approximate kernel row mean by using Nystrom approximation.

    Here, the kernel row mean is the matrix row sum divided by n. Further details for
    Nystrom kernel mean embeddings can be found here :cite:ts:`chatalic2022nystrom`.

    :param key: Key for random number generation
    :param data: The original :math:`n \times d` data
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param num_points: Number of kernel evaluation points
    :return: Approximation of the kernel matrix row sum divided by n
    """
    data = jnp.asarray(data)
    num_data_points = len(data)
    kernel_pairwise = jit(
        vmap(vmap(kernel, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0)
    )
    sample_points = random.choice(key, num_data_points, (num_points,))
    k_mn = kernel_pairwise(data[sample_points], data)
    k_mm = kernel_pairwise(data[sample_points], data[sample_points])
    alpha = (jnp.linalg.pinv(k_mm) @ k_mn).sum(axis=1) / num_data_points

    return k_mn.T @ alpha
