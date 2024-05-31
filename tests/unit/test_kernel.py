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

"""
Tests for kernel implementations.

The tests within this file verify that the implementations of kernels used throughout
the codebase produce the expected results on simple examples.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, Literal, NamedTuple, TypeVar, Union
from unittest.mock import MagicMock

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike
from scipy.stats import norm as scipy_norm
from typing_extensions import override

from coreax.data import Data
from coreax.kernel import (
    AdditiveKernel,
    Kernel,
    LaplacianKernel,
    LinearKernel,
    PairedKernel,
    PCIMQKernel,
    ProductKernel,
    SquaredExponentialKernel,
    SteinKernel,
)

_Kernel = TypeVar("_Kernel", bound=Kernel)
_PairedKernel = TypeVar("_PairedKernel", bound=PairedKernel)


# Once we support only python 3.11+ this should be generic on _Kernel
class _Problem(NamedTuple):
    x: ArrayLike
    y: ArrayLike
    expected_output: ArrayLike
    kernel: Kernel


class BaseKernelTest(ABC, Generic[_Kernel]):
    """Test the ``compute`` methods of a ``coreax.kernel.Kernel``."""

    @abstractmethod
    def kernel(self) -> _Kernel:
        """Abstract pytest fixture which initialises a kernel with parameters fixed."""

    @abstractmethod
    def problem(self, request: pytest.FixtureRequest, kernel: _Kernel) -> _Problem:
        """Abstract pytest fixture which returns a problem for ``Kernel.compute``."""

    def test_compute(
        self, jit_variant: Callable[[Callable], Callable], problem: _Problem
    ) -> None:
        """Test ``compute`` method of ``coreax.kernel.Kernel``."""
        x, y, expected_output, kernel = problem
        output = jit_variant(kernel.compute)(x, y)
        np.testing.assert_array_almost_equal(output, expected_output)


class KernelMeanTest(Generic[_Kernel]):
    """Test the ``compute_mean`` method of a ``coreax.kernel.Kernel``."""

    @pytest.mark.parametrize(
        "block_size",
        [None, 0, -1, 1.2, 2, 3, 9, (None, 2), (2, None)],
        ids=[
            "none",
            "zero",
            "negative",
            "floating",
            "integer_multiple",
            "fractional_multiple",
            "oversized",
            "tuple[none, integer_multiple]",
            "tuple[integer_multiple, none]",
        ],
    )
    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_compute_mean(
        self,
        jit_variant: Callable[[Callable], Callable],
        kernel: _Kernel,
        block_size: Union[int, float, None],
        axis: Union[int, None],
    ) -> None:
        """
        Test the `compute_mean` methods.

        Considers all classes of 'block_size' and 'axis', along with implicitly and
        explicitly weighted data.
        """
        x = jnp.array(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [7.0, 8.0],
            ]
        )
        y = x[:-1] + 1
        kernel_matrix = kernel.compute(x, y)
        x_weights, y_weights = jnp.arange(x.shape[0]), jnp.arange(y.shape[0])
        x_data, y_data = Data(x, x_weights), Data(y, y_weights)
        if axis == 0:
            weights = x_weights
        elif axis == 1:
            weights = y_weights
        else:
            weights = x_weights[..., None] * y_weights[None, ...]
        expected = jnp.average(kernel_matrix, axis, weights)
        test_fn = jit_variant(kernel.compute_mean)
        mean_output = test_fn(x_data, y_data, axis, block_size=block_size)
        np.testing.assert_array_almost_equal(mean_output, expected, decimal=5)

    def test_gramian_row_mean(
        self, jit_variant: Callable[[Callable], Callable], kernel: _Kernel
    ) -> None:
        """Test `gramian_row_mean` behaves as a specialized alias of `compute_mean`."""
        bs = None
        unroll = (1, 1)
        x = jnp.array(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [7.0, 8.0],
            ]
        )
        expected_fn = jit_variant(kernel.compute_mean)
        output_fn = jit_variant(kernel.gramian_row_mean)
        expected = expected_fn(x, x, axis=0, block_size=bs, unroll=unroll)
        output = output_fn(x, block_size=bs, unroll=unroll)
        np.testing.assert_array_equal(output, expected)


class KernelGradientTest(ABC, Generic[_Kernel]):
    """Test the gradient and divergence methods of a ``coreax.kernel.Kernel``."""

    @pytest.fixture(scope="class")
    def gradient_problem(self):
        """Return a problem for testing kernel gradients and divergence."""
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))
        return x, y

    @pytest.mark.parametrize("mode", ["grad_x", "grad_y", "divergence_x_grad_y"])
    @pytest.mark.parametrize("elementwise", [False, True])
    def test_gradients(
        self,
        gradient_problem: tuple[Array, Array],
        kernel: _Kernel,
        mode: Literal["grad_x", "grad_y", "divergence_x_grad_y"],
        elementwise: bool,
    ):
        """Test computation of the kernel gradients."""
        x, y = gradient_problem
        test_mode = mode
        reference_mode = "expected_" + mode
        if elementwise:
            test_mode += "_elementwise"
            x, y = x[:, 0], y[:, 0]
        expected_output = getattr(self, reference_mode)(x, y, kernel)
        if elementwise:
            expected_output = expected_output.squeeze()
        output = getattr(kernel, test_mode)(x, y)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    @abstractmethod
    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: _Kernel
    ) -> Union[Array, np.ndarray]:
        """Compute expected gradient of the kernel w.r.t ``x``."""

    @abstractmethod
    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: _Kernel
    ) -> Union[Array, np.ndarray]:
        """Compute expected gradient of the kernel w.r.t ``y``."""

    @abstractmethod
    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: _Kernel
    ) -> Union[Array, np.ndarray]:
        """Compute expected divergence of the kernel w.r.t ``x`` gradient ``y``."""


class _MockedPairedKernel(Kernel):
    """
    Mock PairedKernel class.

    :param num_points: Size of the mock dataset the kernel will act on
    :param dimension: Dimension of the mock dataset the kernel will act on
    """

    num_points: int = 5
    dimension: int = 5

    first_kernel = MagicMock(spec=Kernel)
    first_kernel.compute_elementwise.return_value = np.array(1.0)
    first_kernel.compute.return_value = np.full((num_points, num_points), 1.0)
    first_kernel.grad_x_elementwise.return_value = np.full(dimension, 1.0)
    first_kernel.grad_x.return_value = np.full((num_points, num_points, dimension), 1.0)
    first_kernel.grad_y_elementwise.return_value = np.full(dimension, 1.0)
    first_kernel.grad_y.return_value = np.full((num_points, num_points, dimension), 1.0)
    first_kernel.divergence_x_grad_y_elementwise.return_value = np.array(1.0)
    first_kernel.divergence_x_grad_y.return_value = np.full(
        (num_points, num_points), 1.0
    )

    second_kernel = MagicMock(spec=Kernel)
    second_kernel.compute_elementwise.return_value = np.array(2.0)
    second_kernel.compute.return_value = np.full((num_points, num_points), 2.0)
    second_kernel.grad_x_elementwise.return_value = np.full(dimension, 2.0)
    second_kernel.grad_x.return_value = np.full(
        (num_points, num_points, dimension), 2.0
    )
    second_kernel.grad_y_elementwise.return_value = np.full(dimension, 2.0)
    second_kernel.grad_y.return_value = np.full(
        (num_points, num_points, dimension), 2.0
    )
    second_kernel.divergence_x_grad_y_elementwise.return_value = np.array(2.0)
    second_kernel.divergence_x_grad_y.return_value = np.full(
        (num_points, num_points), 2.0
    )

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return jnp.array([])

    def reset_mocked_sub_kernels(self) -> None:
        """Reset the mocked sub-kernels."""
        self.first_kernel.reset_mock()
        self.second_kernel.reset_mock()


class TestAdditiveKernel(
    BaseKernelTest[_MockedPairedKernel],
    KernelMeanTest[_MockedPairedKernel],
    KernelGradientTest[_MockedPairedKernel],
):
    """Test ``coreax.kernel.AdditiveKernel``."""

    # Set size and dimension of mock "dataset" that the mocked kernel will act on
    mock_num_points = 5
    mock_dimension = 2

    @pytest.fixture(scope="class")
    def kernel(self) -> _MockedPairedKernel:
        """Return a mocked paired kernel function."""
        return _MockedPairedKernel(
            num_points=self.mock_num_points, dimension=self.mock_dimension
        )

    def to_additive_kernel(
        self, kernel: _MockedPairedKernel
    ) -> tuple[_MockedPairedKernel, AdditiveKernel]:
        """Construct an Additive kernel using reset mocked sub kernels."""
        kernel.reset_mocked_sub_kernels()
        return kernel, AdditiveKernel(kernel.first_kernel, kernel.second_kernel)

    @pytest.fixture(params=["floats", "vectors", "arrays"])
    def problem(self, request, kernel: _MockedPairedKernel) -> _Problem:
        r"""
        Test problems for the Additive kernel.

        Given kernel functions :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`
        and :math:`l:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`, define the
        additive kernel :math:`p:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` where
        :math:`p(x,y) := k(x,y) + l(x,y)`

        We consider the simplest possible example of adding two Linear kernels together
        with the following cases:
        - `floats`: where x and y are floats
        - `vectors`: where x and y are vectors of the same size
        - `arrays`: where x and y are arrays of the same shape
        """
        mode = request.param
        x = 0.5
        y = 2.0
        if mode == "floats":
            expected_distances = 2.0
        elif mode == "vectors":
            x = 1.0 * np.arange(4)
            y = x + 1.0
            expected_distances = 40.0
        elif mode == "arrays":
            x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
            y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
            expected_distances = np.array([[40, 88], [140, 348]])
        else:
            raise ValueError("Invalid problem mode")
        output_scale = 1.0
        constant = 0.0

        # Extract additive kernel and replace mocked kernels with actual kernels
        _, additive_kernel = self.to_additive_kernel(kernel)
        modified_kernel = eqx.tree_at(
            lambda x: x.second_kernel,
            eqx.tree_at(
                lambda x: x.first_kernel,
                additive_kernel,
                LinearKernel(output_scale, constant),
            ),
            LinearKernel(output_scale, constant),
        )
        return _Problem(x, y, expected_distances, modified_kernel)

    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: _MockedPairedKernel
    ) -> np.ndarray:
        _, additive_kernel = self.to_additive_kernel(kernel)
        num_points, dimension = np.atleast_2d(x).shape

        grad_1 = additive_kernel.first_kernel.grad_x_elementwise(x, y)
        grad_2 = additive_kernel.second_kernel.grad_x_elementwise(x, y)
        expected_grad = grad_1 + grad_2

        shape = num_points, num_points, kernel.dimension
        if dimension != 1:
            expected_grad = np.tile(expected_grad, num_points**2).reshape(shape)

        return expected_grad

    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: _MockedPairedKernel
    ) -> np.ndarray:
        _, additive_kernel = self.to_additive_kernel(kernel)
        num_points, dimension = np.atleast_2d(x).shape

        grad_1 = additive_kernel.first_kernel.grad_y_elementwise(x, y)
        grad_2 = additive_kernel.second_kernel.grad_y_elementwise(x, y)
        expected_grad = grad_1 + grad_2

        shape = num_points, num_points, kernel.dimension
        if dimension != 1:
            expected_grad = np.tile(expected_grad, num_points**2).reshape(shape)

        return expected_grad

    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: _MockedPairedKernel
    ) -> np.ndarray:
        _, additive_kernel = self.to_additive_kernel(kernel)
        num_points, _ = np.atleast_2d(x).shape

        expected_divergences = np.tile(
            additive_kernel.first_kernel.divergence_x_grad_y_elementwise(x, y)
            + additive_kernel.second_kernel.divergence_x_grad_y_elementwise(x, y),
            num_points**2,
        ).reshape(num_points, num_points)

        return expected_divergences

    def test_kernel_calls_sub_kernels_correctly(
        self, kernel: _MockedPairedKernel
    ) -> None:
        """Ensure AdditiveKernel calls sub kernel methods correctly."""
        x = np.array(1.0)
        y = np.array(1.0)

        # Reset the mocked kernels and build an additive kernel using them; variable
        # rename allows for nicer automatic formatting.
        k, additive_kernel = self.to_additive_kernel(kernel)

        additive_kernel.compute_elementwise(x, y)
        k.first_kernel.compute_elementwise.assert_called_once_with(x, y)
        k.second_kernel.compute_elementwise.assert_called_once_with(x, y)

        k, additive_kernel = self.to_additive_kernel(k)
        additive_kernel.grad_x_elementwise(x=x, y=x)
        k.first_kernel.grad_x_elementwise.assert_called_once_with(x, y)
        k.second_kernel.grad_x_elementwise.assert_called_once_with(x, y)

        k, additive_kernel = self.to_additive_kernel(k)
        additive_kernel.grad_y_elementwise(x=x, y=x)
        k.first_kernel.grad_y_elementwise.assert_called_once_with(x, y)
        k.second_kernel.grad_y_elementwise.assert_called_once_with(x, y)

        k, additive_kernel = self.to_additive_kernel(k)
        additive_kernel.divergence_x_grad_y_elementwise(x, y)
        k.first_kernel.divergence_x_grad_y_elementwise.assert_called_once_with(x, y)
        k.second_kernel.divergence_x_grad_y_elementwise.assert_called_once_with(x, y)


class TestProductKernel(
    BaseKernelTest[_MockedPairedKernel],
    KernelMeanTest[_MockedPairedKernel],
    KernelGradientTest[_MockedPairedKernel],
):
    """Test ``coreax.kernel.ProductKernel``."""

    # Set size and dimension of mock "dataset" that the mocked kernel will act on
    mock_num_points = 5
    mock_dimension = 2

    @pytest.fixture(scope="class")
    def kernel(self) -> _MockedPairedKernel:
        """Return a mocked paired kernel function."""
        return _MockedPairedKernel(
            num_points=self.mock_num_points, dimension=self.mock_dimension
        )

    def to_product_kernel(
        self, kernel: _MockedPairedKernel
    ) -> tuple[_MockedPairedKernel, ProductKernel]:
        """Construct a Product kernel using reset mocked sub kernels."""
        kernel.reset_mocked_sub_kernels()
        return kernel, ProductKernel(kernel.first_kernel, kernel.second_kernel)

    @pytest.fixture(params=["floats", "vectors", "arrays"])
    def problem(self, request, kernel: _MockedPairedKernel) -> _Problem:
        r"""
        Test problems for the Product kernel where we multiply two Linear kernels.

        Given kernel functions :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`
        and :math:`l:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`, define the
        product kernel :math:`p:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` where
        :math:`p(x,y) = k(x,y)l(x,y)`

        We consider the simplest possible example of adding two Linear kernels together
        with the following cases:
        - `floats`: where x and y are floats
        - `vectors`: where x and y are vectors of the same size
        - `arrays`: where x and y are arrays of the same shape
        """
        mode = request.param
        x = 1.5
        y = 2.0
        if mode == "floats":
            expected_distances = 9.0
        elif mode == "vectors":
            x = 1.0 * np.arange(4)
            y = x + 1.0
            expected_distances = 400.0
        elif mode == "arrays":
            x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
            y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
            expected_distances = np.array([[400, 1936], [4900, 30276]])
        else:
            raise ValueError("Invalid problem mode")
        output_scale = 1.0
        constant = 0.0

        # Extract product kernel and replace mocked kernels with actual kernels
        _, product_kernel = self.to_product_kernel(kernel)
        modified_kernel = eqx.tree_at(
            lambda x: x.second_kernel,
            eqx.tree_at(
                lambda x: x.first_kernel,
                product_kernel,
                LinearKernel(output_scale, constant),
            ),
            LinearKernel(output_scale, constant),
        )
        return _Problem(x, y, expected_distances, modified_kernel)

    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: _MockedPairedKernel
    ) -> np.ndarray:
        num_points, dimension = np.atleast_2d(x).shape

        grad_1 = kernel.first_kernel.grad_x_elementwise(x, y)
        compute_1 = kernel.first_kernel.compute_elementwise(x, y)
        grad_2 = kernel.second_kernel.grad_x_elementwise(x, y)
        compute_2 = kernel.second_kernel.compute_elementwise(x, y)
        expected_grad = grad_1 * compute_2 + grad_2 * compute_1

        shape = num_points, num_points, kernel.dimension
        if dimension != 1:
            expected_grad = np.tile(expected_grad, num_points**2).reshape(shape)

        return expected_grad

    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: _MockedPairedKernel
    ) -> np.ndarray:
        num_points, dimension = np.atleast_2d(x).shape

        grad_1 = kernel.first_kernel.grad_y_elementwise(x, y)
        compute_1 = kernel.first_kernel.compute_elementwise(x, y)
        grad_2 = kernel.second_kernel.grad_y_elementwise(x, y)
        compute_2 = kernel.second_kernel.compute_elementwise(x, y)
        expected_grad = grad_1 * compute_2 + grad_2 * compute_1

        shape = num_points, num_points, kernel.dimension
        if dimension != 1:
            expected_grad = np.tile(expected_grad, num_points**2).reshape(shape)

        return expected_grad

    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: _MockedPairedKernel
    ) -> np.ndarray:
        # Variable rename allows for nicer automatic formatting
        k1, k2 = kernel.first_kernel, kernel.second_kernel
        expected_divergences = (
            k1.grad_x_elementwise(x, y).dot(k2.grad_y_elementwise(x, y))
            + k1.grad_y_elementwise(x, y).dot(k2.grad_x_elementwise(x, y))
            + k1.compute_elementwise(x, y) * k2.divergence_x_grad_y_elementwise(x, y)
            + k2.compute_elementwise(x, y) * k1.divergence_x_grad_y_elementwise(x, y)
        )

        return expected_divergences

    def test_kernel_calls_sub_kernels_correctly(
        self, kernel: _MockedPairedKernel
    ) -> None:
        """Ensure ProductKernel calls sub kernel methods correctly."""
        x = np.array(1.0)
        y = np.array(1.0)

        # Reset the mocked kernels and build an product kernel using them; variable
        # rename allows for nicer automatic formatting
        k, product_kernel = self.to_product_kernel(kernel)

        product_kernel.compute_elementwise(x, y)
        k.first_kernel.compute_elementwise.assert_called_once_with(x, y)
        k.second_kernel.compute_elementwise.assert_called_once_with(x, y)

        k, product_kernel = self.to_product_kernel(kernel)
        product_kernel.grad_x_elementwise(x=x, y=x)
        k.first_kernel.compute_elementwise.assert_called_once_with(x, y)
        k.second_kernel.compute_elementwise.assert_called_once_with(x, y)
        k.first_kernel.grad_x_elementwise.assert_called_once_with(x, y)
        k.second_kernel.grad_x_elementwise.assert_called_once_with(x, y)

        k, product_kernel = self.to_product_kernel(kernel)
        product_kernel.grad_y_elementwise(x=x, y=x)
        k.first_kernel.compute_elementwise.assert_called_once_with(x, y)
        k.second_kernel.compute_elementwise.assert_called_once_with(x, y)
        k.first_kernel.grad_y_elementwise.assert_called_once_with(x, y)
        k.second_kernel.grad_y_elementwise.assert_called_once_with(x, y)

        k, product_kernel = self.to_product_kernel(kernel)
        product_kernel.divergence_x_grad_y_elementwise(x, y)
        k.first_kernel.compute_elementwise.assert_called_once_with(x, y)
        k.second_kernel.compute_elementwise.assert_called_once_with(x, y)
        k.first_kernel.grad_x_elementwise.assert_called_once_with(x, y)
        k.second_kernel.grad_x_elementwise.assert_called_once_with(x, y)
        k.first_kernel.grad_y_elementwise.assert_called_once_with(x, y)
        k.second_kernel.grad_y_elementwise.assert_called_once_with(x, y)
        k.first_kernel.divergence_x_grad_y_elementwise.assert_called_once_with(x, y)
        k.second_kernel.divergence_x_grad_y_elementwise.assert_called_once_with(x, y)


class TestLinearKernel(
    BaseKernelTest[LinearKernel],
    KernelMeanTest[LinearKernel],
    KernelGradientTest[LinearKernel],
):
    """Test ``coreax.kernel.LinearKernel``."""

    @pytest.fixture(scope="class")
    def kernel(self) -> LinearKernel:
        random_seed = 2_024
        output_scale, constant = jnp.abs(jr.normal(key=jr.key(random_seed), shape=(2,)))
        return LinearKernel(output_scale, constant)

    @pytest.fixture(params=["floats", "vectors", "arrays"])
    def problem(self, request: pytest.FixtureRequest, kernel: LinearKernel) -> _Problem:
        r"""
        Test problems for the Linear kernel.

        The kernel is defined as
        :math:`k(x,y) = \text{output_scale} * x^Ty` + \text{constant}.

        We consider the following cases:
        - `floats`: where x and y are floats
        - `vectors`: where x and y are vectors of the same size
        - `arrays`: where x and y are arrays of the same shape
        """
        mode = request.param
        x = 0.5
        y = 2.0
        if mode == "floats":
            expected_distances = 1.0
        elif mode == "vectors":
            x = 1.0 * np.arange(4)
            y = x + 1.0
            expected_distances = 20.0
        elif mode == "arrays":
            x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
            y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
            expected_distances = np.array([[20, 44], [70, 174]])
        else:
            raise ValueError("Invalid problem mode")
        modified_kernel = eqx.tree_at(
            lambda x: x.output_scale, eqx.tree_at(lambda x: x.constant, kernel, 0), 1.0
        )
        return _Problem(x, y, expected_distances, modified_kernel)

    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: LinearKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_gradients[x_idx, y_idx] = kernel.output_scale * y[y_idx]
        return expected_gradients

    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: LinearKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_gradients[x_idx, y_idx] = kernel.output_scale * x[x_idx]
        return expected_gradients

    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: LinearKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_divergence = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_divergence[x_idx, y_idx] = kernel.output_scale * dimension
        return expected_divergence


class TestSquaredExponentialKernel(
    BaseKernelTest[SquaredExponentialKernel],
    KernelMeanTest[SquaredExponentialKernel],
    KernelGradientTest[SquaredExponentialKernel],
):
    """Test ``coreax.kernel.SquaredExponentialKernel``."""

    @pytest.fixture(scope="class")
    def kernel(self) -> SquaredExponentialKernel:
        random_seed = 2_024
        length_scale, output_scale = jnp.abs(
            jr.normal(key=jr.key(random_seed), shape=(2,))
        )
        return SquaredExponentialKernel(length_scale, output_scale)

    @pytest.fixture(
        params=[
            "floats",
            "vectors",
            "arrays",
            "normalized",
            "negative_length_scale",
            "large_negative_length_scale",
            "near_zero_length_scale",
            "negative_output_scale",
        ]
    )
    def problem(  # noqa: C901
        self, request: pytest.FixtureRequest, kernel: SquaredExponentialKernel
    ) -> _Problem:
        r"""
        Test problems for the SquaredExponential kernel.

        The kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||^2/(2 * \text{length_scale}^2))`.

        We consider the following cases:
        1. length scale is :math:`\sqrt{\pi} / 2`:
        - `floats`: where x and y are floats
        - `vectors`: where x and y are vectors of the same size
        - `arrays`: where x and y are arrays of the same shape

        2. length scale is :math:`\exp(1)` and output scale is
        :math:`\frac{1}{\sqrt{2*\pi} * \exp(1)}`:
        - `normalized`: where x and y are vectors of the same size (this is the
        special case where the squared exponential kernel is the Gaussian kernel)

        3. length scale or output scale is degenerate:
        - `negative_length_scale`: should give same result as positive equivalent
        - `large_negative_length_scale`: should approximately equal one
        - `near_zero_length_scale`: should approximately equal zero
        - `negative_output_scale`: should negate the positive equivalent.
        """
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)
        output_scale = 1.0
        mode = request.param
        x = 0.5
        y = 2.0
        if mode == "floats":
            expected_distances = 0.48860678
        elif mode == "vectors":
            x = 1.0 * np.arange(4)
            y = x + 1.0
            expected_distances = 0.279923327
        elif mode == "arrays":
            x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
            y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
            expected_distances = np.array(
                [[0.279923327, 1.4996075e-14], [1.4211038e-09, 1.0]]
            )
        elif mode == "normalized":
            length_scale = np.e
            output_scale = 1 / (np.sqrt(2 * np.pi) * length_scale)
            num_points = 10
            x = np.arange(num_points)
            y = x + 1.0

            # Compute expected output using standard implementation of the Gaussian PDF
            expected_distances = np.zeros((num_points, num_points))
            for x_idx, x_ in enumerate(x):
                for y_idx, y_ in enumerate(y):
                    expected_distances[x_idx, y_idx] = scipy_norm(y_, length_scale).pdf(
                        x_
                    )
            x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        elif mode == "negative_length_scale":
            length_scale = -1
            expected_distances = 0.324652467
        elif mode == "large_negative_length_scale":
            length_scale = -10000.0
            expected_distances = 1.0
        elif mode == "near_zero_length_scale":
            length_scale = -0.0000001
            expected_distances = 0.0
        elif mode == "negative_output_scale":
            length_scale = 1
            output_scale = -1
            expected_distances = -0.324652467
        else:
            raise ValueError("Invalid problem mode")
        modified_kernel = eqx.tree_at(
            lambda x: x.output_scale,
            eqx.tree_at(lambda x: x.length_scale, kernel, length_scale),
            output_scale,
        )
        return _Problem(x, y, expected_distances, modified_kernel)

    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: SquaredExponentialKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_gradients[x_idx, y_idx] = (
                    -kernel.output_scale
                    * (x[x_idx, :] - y[y_idx, :])
                    / kernel.length_scale**2
                    * np.exp(
                        -(np.linalg.norm(x[x_idx, :] - y[y_idx, :]) ** 2)
                        / (2 * kernel.length_scale**2)
                    )
                )
        return expected_gradients

    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: SquaredExponentialKernel
    ) -> np.ndarray:
        return -self.expected_grad_x(x, y, kernel)

    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: SquaredExponentialKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_divergence = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                dot_product = np.dot(
                    x[x_idx, :] - y[y_idx, :], x[x_idx, :] - y[y_idx, :]
                )
                kernel_scaled = kernel.output_scale * np.exp(
                    -dot_product / (2.0 * kernel.length_scale**2)
                )
                expected_divergence[x_idx, y_idx] = (
                    kernel_scaled
                    / kernel.length_scale**2
                    * (dimension - dot_product / kernel.length_scale**2)
                )
        return expected_divergence


class TestLaplacianKernel(
    BaseKernelTest[LaplacianKernel],
    KernelMeanTest[LaplacianKernel],
    KernelGradientTest[LaplacianKernel],
):
    """Test ``coreax.kernel.LaplacianKernel``."""

    @pytest.fixture(scope="class")
    def kernel(self) -> LaplacianKernel:
        random_seed = 2_024
        length_scale, output_scale = jnp.abs(
            jr.normal(key=jr.key(random_seed), shape=(2,))
        )
        return LaplacianKernel(length_scale, output_scale)

    @pytest.fixture(
        params=[
            "floats",
            "vectors",
            "arrays",
            "negative_length_scale",
            "large_negative_length_scale",
            "near_zero_length_scale",
            "negative_output_scale",
        ]
    )
    def problem(
        self, request: pytest.FixtureRequest, kernel: LaplacianKernel
    ) -> _Problem:
        r"""
        Test problems for the Laplacian kernel.

        The kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||_1/(2 * \text{length_scale}^2))`.

        We consider the following cases:
        1. length scale is :math:`\sqrt{\pi} / 2`:
        - `floats`: where x and y are floats
        - `vectors`: where x and y are vectors of the same size
        - `arrays`: where x and y are arrays of the same shape

        2. length scale or output scale is degenerate:
        - `negative_length_scale`: should give same result as positive equivalent
        - `large_negative_length_scale`: should approximately equal one
        - `near_zero_length_scale`: should approximately equal zero
        - `negative_output_scale`: should negate the positive equivalent.
        """
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)
        output_scale = 1.0
        mode = request.param
        x = 0.5
        y = 2.0
        if mode == "floats":
            expected_distances = 0.62035410351
        elif mode == "vectors":
            x = 1.0 * np.arange(4)
            y = x + 1.0
            expected_distances = 0.279923327
        elif mode == "arrays":
            x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
            y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
            expected_distances = np.array(
                [[0.279923327, 0.00171868172], [0.00613983027, 1.0]]
            )
        elif mode == "negative_length_scale":
            length_scale = -1
            expected_distances = 0.472366553
        elif mode == "large_negative_length_scale":
            length_scale = -10000.0
            expected_distances = 1.0
        elif mode == "near_zero_length_scale":
            length_scale = -0.0000001
            expected_distances = 0.0
        elif mode == "negative_output_scale":
            length_scale = 1
            output_scale = -1
            expected_distances = -0.472366553
        else:
            raise ValueError("Invalid problem mode")
        modified_kernel = eqx.tree_at(
            lambda x: x.output_scale,
            eqx.tree_at(lambda x: x.length_scale, kernel, length_scale),
            output_scale,
        )
        return _Problem(x, y, expected_distances, modified_kernel)

    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: LaplacianKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_gradients[x_idx, y_idx] = (
                    -kernel.output_scale
                    * np.sign(x[x_idx, :] - y[y_idx, :])
                    / (2 * kernel.length_scale**2)
                    * np.exp(
                        -np.linalg.norm(x[x_idx, :] - y[y_idx, :], ord=1)
                        / (2 * kernel.length_scale**2)
                    )
                )
        return expected_gradients

    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: LaplacianKernel
    ) -> np.ndarray:
        return -self.expected_grad_x(x, y, kernel)

    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: LaplacianKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_divergence = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_divergence[x_idx, y_idx] = (
                    -kernel.output_scale
                    * dimension
                    / (4 * kernel.length_scale**4)
                    * np.exp(
                        -np.linalg.norm(x[x_idx, :] - y[y_idx, :], ord=1)
                        / (2 * kernel.length_scale**2)
                    )
                )
        return expected_divergence


class TestPCIMQKernel(
    BaseKernelTest[PCIMQKernel],
    KernelMeanTest[PCIMQKernel],
    KernelGradientTest[PCIMQKernel],
):
    """Test ``coreax.kernel.PCIMQKernel``."""

    @pytest.fixture(scope="class")
    def kernel(self) -> PCIMQKernel:
        random_seed = 2_024
        length_scale, output_scale = jnp.abs(
            jr.normal(key=jr.key(random_seed), shape=(2,))
        )
        return PCIMQKernel(length_scale, output_scale)

    @pytest.fixture(
        params=[
            "floats",
            "vectors",
            "arrays",
            "negative_length_scale",
            "large_negative_length_scale",
            "near_zero_length_scale",
            "negative_output_scale",
        ]
    )
    def problem(self, request: pytest.FixtureRequest, kernel: PCIMQKernel) -> _Problem:
        r"""
        Test problems for the PCIMQ kernel.

        The kernel is defined as
        :math:`k(x,y) = \frac{1}{\sqrt(1 + ((x - y) / \text{length_scale}) ** 2 / 2)}`.

        We consider the following cases:
        1. length scale is :math:`\sqrt{\pi} / 2`:
        - `floats`: where x and y are floats
        - `vectors`: where x and y are vectors of the same size
        - `arrays`: where x and y are arrays of the same shape

        2. length scale or output scale is degenerate:
        - `negative_length_scale`: should give same result as positive equivalent
        - `large_negative_length_scale`: should approximately equal one
        - `near_zero_length_scale`: should approximately equal zero
        - `negative_output_scale`: should negate the positive equivalent.
        """
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)
        output_scale = 1.0
        mode = request.param
        x = 0.5
        y = 2.0
        if mode == "floats":
            expected_distances = 0.76333715144
        elif mode == "vectors":
            x = 1.0 * np.arange(4)
            y = x + 1.0
            expected_distances = 0.66325021409
        elif mode == "arrays":
            x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
            y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
            expected_distances = np.array(
                [[0.66325021409, 0.17452514991], [0.21631125495, 1.0]]
            )
        elif mode == "negative_length_scale":
            length_scale = -1
            expected_distances = 0.685994341
        elif mode == "large_negative_length_scale":
            length_scale = -10000.0
            expected_distances = 1.0
        elif mode == "near_zero_length_scale":
            length_scale = -0.0000001
            expected_distances = 0.0
        elif mode == "negative_output_scale":
            length_scale = 1
            output_scale = -1
            expected_distances = -0.685994341
        else:
            raise ValueError("Invalid problem mode")
        modified_kernel = eqx.tree_at(
            lambda x: x.output_scale,
            eqx.tree_at(lambda x: x.length_scale, kernel, length_scale),
            output_scale,
        )
        return _Problem(x, y, expected_distances, modified_kernel)

    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: PCIMQKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                scaling = 2 * kernel.length_scale**2
                mq_array = np.linalg.norm(x[x_idx, :] - y[y_idx, :]) ** 2 / scaling
                primal = kernel.output_scale / np.sqrt(1 + mq_array)
                expected_gradients[x_idx, y_idx] = (
                    -kernel.output_scale
                    / scaling
                    * (x[x_idx, :] - y[y_idx, :])
                    * (primal / kernel.output_scale) ** 3
                )
        return expected_gradients

    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: PCIMQKernel
    ) -> np.ndarray:
        return -self.expected_grad_x(x, y, kernel)

    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: PCIMQKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        length_scale = kernel.length_scale
        output_scale = kernel.output_scale
        expected_divergence = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                dot_product = np.dot(
                    x[x_idx, :] - y[y_idx, :], x[x_idx, :] - y[y_idx, :]
                )
                kernel_scaled = output_scale / (
                    (1 + dot_product / (2 * length_scale**2)) ** (1 / 2)
                )
                expected_divergence[x_idx, y_idx] = (
                    output_scale
                    / (2 * length_scale**2)
                    * (
                        dimension * (kernel_scaled / output_scale) ** 3
                        - 3
                        * dot_product
                        * (kernel_scaled / output_scale) ** 5
                        / (2 * length_scale**2)
                    )
                )

        return expected_divergence


class TestSteinKernel(BaseKernelTest[SteinKernel]):
    """Test ``coreax.kernel.SteinKernel``."""

    @pytest.fixture(scope="class")
    def kernel(self) -> SteinKernel:
        random_seed = 2_024
        length_scale, output_scale = jnp.abs(
            jr.normal(key=jr.key(random_seed), shape=(2,))
        )
        base_kernel = PCIMQKernel(length_scale, output_scale)
        return SteinKernel(base_kernel=base_kernel, score_function=jnp.negative)

    @pytest.fixture
    def problem(self, request: pytest.FixtureRequest, kernel: SteinKernel) -> _Problem:
        """Test problem for the Stein kernel."""
        length_scale = 1 / np.sqrt(2)
        modified_kernel = eqx.tree_at(
            lambda x: x.base_kernel, kernel, PCIMQKernel(length_scale, 1)
        )
        score_function = modified_kernel.score_function
        beta = 0.5
        num_points_x = 10
        num_points_y = 5
        dimension = 2
        generator = np.random.default_rng(1_989)
        x = generator.random((num_points_x, dimension))
        y = generator.random((num_points_y, dimension))

        def k_x_y(x_input, y_input):
            r"""
            Compute Stein kernel.

            Throughout this docstring, x_input and y_input are simply referred to as x
            and y.

            The base kernel is :math:`(1 + \lvert \mathbf{x} - \mathbf{y}
            \rvert^2)^{-1/2}`. :math:`\mathbb{P}` is :math:`\mathcal{N}(0, \mathbf{I})`
            with :math:`\nabla \log f_X(\mathbf{x}) = -\mathbf{x}`.

            In the code: n, m and r refer to shared denominators in the Stein kernel
            equation (rather than divergence, x_, y_ and z in the main code function).

            :math:`k_\mathbb{P}(\mathbf{x}, \mathbf{y}) = n + m + r`.

            :math:`n := -\frac{3 \lvert \mathbf{x} - \mathbf{y} \rvert^2}{(1 + \lvert
            \mathbf{x} - \mathbf{y} \rvert^2)^{5/2}}`.

            :math:`m := 2\beta\left[ \frac{d + [\mathbf{y} -
            \mathbf{x}]^\intercal[\mathbf{x} - \mathbf{y}]}{(1 + \lvert \mathbf{x} -
            \mathbf{y} \rvert^2)^{3/2}} \right]`.

            :math:`r := \frac{\mathbf{x}^\intercal \mathbf{y}}{(1 + \lvert \mathbf{x} -
            \mathbf{y} \rvert^2)^{1/2}}`.

            :param x_input: a d-dimensional vector
            :param y_input: a d-dimensional vector
            :return: kernel evaluated at x, y
            """
            norm_sq = np.linalg.norm(x_input - y_input) ** 2
            n = -3 * norm_sq / (1 + norm_sq) ** 2.5
            dot_prod = np.dot(
                score_function(x_input) - score_function(y_input),
                x_input - y_input,
            )
            m = 2 * beta * (dimension + dot_prod) / (1 + norm_sq) ** 1.5
            r = (
                np.dot(score_function(x_input), score_function(y_input))
                / (1 + norm_sq) ** 0.5
            )
            return n + m + r

        expected_output = np.zeros([x.shape[0], y.shape[0]])
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                # Compute via our hand-coded kernel evaluation
                expected_output[x_idx, y_idx] = k_x_y(x[x_idx, :], y[y_idx, :])

        return _Problem(x, y, expected_output, modified_kernel)
