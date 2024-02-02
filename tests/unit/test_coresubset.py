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
Tests for coresubset construction approaches.

Coresubsets are coresets in which elements in the coreset must also be elements in the
original dataset. The tests within this file verify that approaches to constructing
coresubsets produce the expected results on simple examples.
"""

import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
from jax import random
from jax.typing import ArrayLike

import coreax.approximation
import coreax.coresubset
import coreax.data
import coreax.kernel
import coreax.metrics
import coreax.reduction
import coreax.refine


class TestKernelHerding(unittest.TestCase):
    """
    Tests related to the KernelHerding class defined in coresubset.py.
    """

    def setUp(self):
        """
        Generate data for use across unit tests.
        """
        # Define data parameters
        self.dimension = 3
        self.random_data_generation_key = 0
        self.coreset_size = 20

    def test_tree_flatten(self) -> None:
        """
        Test that the pytree is flattened as expected.
        """
        # Create a kernel herding object
        kernel = coreax.kernel.SquaredExponentialKernel()
        coresubset_object_herding = coreax.coresubset.KernelHerding(
            kernel=kernel,
        )

        # Set attributes on the object to ensure actual values are returned
        coresubset_object_herding.kernel_matrix_row_sum_mean = None
        coresubset_object_herding.coreset_indices = jnp.zeros(1, dtype=jnp.int32)
        coresubset_object_herding.coreset = jnp.zeros([2, 3])
        coresubset_object_herding.block_size = 5
        coresubset_object_herding.unique = False
        coresubset_object_herding.refine_method = "ABC"
        coresubset_object_herding.weights_optimiser = "DEF"
        coresubset_object_herding.approximator = "XYZ"
        coresubset_object_herding.random_key = 1989

        # Call the method and check each output are as expected
        output_children, output_aux_data = coresubset_object_herding.tree_flatten()

        self.assertEqual(len(output_children), 4)
        self.assertEqual(output_children[0], kernel)
        self.assertIsNone(output_children[1])
        np.testing.assert_array_equal(output_children[2], jnp.zeros(1, dtype=jnp.int32))
        np.testing.assert_array_equal(output_children[3], jnp.zeros([2, 3]))
        self.assertDictEqual(
            output_aux_data,
            {
                "block_size": 5,
                "unique": False,
                "refine_method": "ABC",
                "weights_optimiser": "DEF",
                "approximator": "XYZ",
                "random_key": 1989,
            },
        )

    def test_fit_comparison_to_random_and_refined(self) -> None:
        """
        Test the fit method of the KernelHerding class with a simple example.

        The test checks that a coreset generated via kernel herding has an improved
        quality (measured by maximum mean discrepancy) than one generated by random
        sampling. We further check that if the coreset generated by kernel herding is
        refined, the quality improves yet again.
        """
        # Define specific test instance setup
        kernel = coreax.kernel.SquaredExponentialKernel()
        num_data_points = 100

        # Define some data - sufficiently large that we would not expect a random sample
        # to typically compete with a kernel herding approach
        generator = np.random.default_rng(self.random_data_generation_key)
        x = generator.random((num_data_points, self.dimension))
        data = coreax.data.ArrayData.load(x)

        # Create a kernel herding object
        coresubset_object_herding = coreax.coresubset.KernelHerding(
            kernel=kernel, refine_method=coreax.refine.RefineRegular()
        )

        # Apply kernel herding on the dataset, and record the coreset for comparison
        coresubset_object_herding.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        herding_coreset = coresubset_object_herding.coreset

        # Create a random refinement object and generate a coreset, for comparison
        coresubset_object_random = coreax.coresubset.RandomSample(
            random_seed=0, unique=True
        )
        coresubset_object_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        random_coreset = coresubset_object_random.coreset

        # Define a metric and compare quality of produced coresets
        metric = coreax.metrics.MMD(kernel=kernel)
        herding_metric = metric.compute(x, herding_coreset)
        random_metric = metric.compute(x, random_coreset)
        self.assertLess(float(herding_metric), float(random_metric))

        # Create a coreset via kernel herding with refinement, to check refinement
        # improves the coreset quality
        coresubset_object_herding.refine()
        refined_herding_coreset = coresubset_object_herding.coreset

        # Compare quality of refined coreset to non-refined coreset
        refined_herding_metric = metric.compute(x, refined_herding_coreset)
        self.assertLess(float(refined_herding_metric), float(herding_metric))

    def test_fit_compare_row_sum(self) -> None:
        """
        Test the fit method of the KernelHerding class handling the kernel row sum mean.

        The test checks that when the kernel matrix row sum mean is passed, and not
        passed, the same answer is produced by the herding algorithm.
        """
        # Define specific test instance setup
        kernel = coreax.kernel.LaplacianKernel()
        num_data_points = 100

        # Define some data
        generator = np.random.default_rng(self.random_data_generation_key)
        x = generator.random((num_data_points, self.dimension))
        data = coreax.data.ArrayData.load(x)

        # Create a kernel herding object
        coresubset_object_herding = coreax.coresubset.KernelHerding(kernel=kernel)

        # Apply kernel herding on the dataset, and record the coreset for comparison
        coresubset_object_herding.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        herding_coreset = coresubset_object_herding.coreset

        # Compute the kernel matrix row sum mean outside of the herding object
        kernel_matrix_row_sum = kernel.calculate_kernel_matrix_row_sum_mean(x=x)
        coresubset_object_herding.kernel_matrix_row_sum_mean = kernel_matrix_row_sum
        coresubset_object_herding.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        herding_coreset_pre_computed_mean = coresubset_object_herding.coreset

        # Check the two coresets agree
        np.testing.assert_array_equal(
            herding_coreset, herding_coreset_pre_computed_mean
        )

        # The previous check ensures that the result is the same, however we need to
        # test the passed kernel matrix row sum is being used. To do this, we give an
        # incorrect random kernel matrix row sum and check the resulting coreset is
        # different.
        coresubset_object_herding.kernel_matrix_row_sum_mean = (
            0.5 * kernel_matrix_row_sum
        )
        coresubset_object_herding.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        herding_coreset_invalid_mean = coresubset_object_herding.coreset
        coreset_difference = abs(herding_coreset - herding_coreset_invalid_mean)
        self.assertGreater(coreset_difference.sum(), 0)

    # pylint: disable=too-many-locals
    def test_fit_comparison_to_random_stein(self) -> None:
        """
        Test the fit method of the KernelHerding class with a Stein kernel.

        The test checks that a coreset generated via kernel herding has an improved
        quality (measured by maximum mean discrepancy) than one generated by random
        sampling. A Stein kernel is used, which adds further complication to kernel
        evaluation. All of this complication should be handled inside the kernel itself,
        and the user not have to alter their usage after the kernel is defined.
        """
        # Define specific test instance setup. A uni-variate Gaussian is used as it has
        # a known score function to pass to the Stein kernel.
        mu = 0.0
        std_dev = 1.0
        num_data_points = 500
        generator = np.random.default_rng(1_989)
        x = generator.normal(mu, std_dev, size=(num_data_points, 1))

        def true_score(x_: ArrayLike) -> ArrayLike:
            """
            Compute the known score function of a uni-variate Gaussian.

            :param x_: Point at which we want to evaluate the score function
            :return: Score function, evaluated at point of interest
            """
            return -(x_ - mu) / std_dev**2

        # Create kernel and data objects
        kernel = coreax.kernel.SteinKernel(
            base_kernel=coreax.kernel.PCIMQKernel(length_scale=0.5),
            score_function=true_score,
        )
        data = coreax.data.ArrayData.load(x)

        # Create a kernel herding object
        coresubset_object_herding = coreax.coresubset.KernelHerding(kernel=kernel)

        # Apply kernel herding on the dataset, and record the coreset for comparison
        coresubset_object_herding.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        herding_coreset = coresubset_object_herding.coreset

        # Create a random refinement object and generate a coreset, for comparison
        coresubset_object_random = coreax.coresubset.RandomSample(
            weights_optimiser=None, kernel=kernel
        )
        coresubset_object_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        random_coreset = coresubset_object_random.coreset

        # Define a metric and compare quality of produced coresets
        metric = coreax.metrics.MMD(kernel=kernel)
        herding_metric = metric.compute(x, herding_coreset)
        random_metric = metric.compute(x, random_coreset)
        self.assertLess(float(herding_metric), float(random_metric))

    def test_fit_with_approximate_kernel_matrix_row_sum_mean(self):
        """
        Test fit in the KernelHerding class when approximating the kernel row sum mean.

        The test checks that the approximation method is called exactly once. It also
        checks that a coreset generated via kernel herding has an improved quality
        (measured by maximum mean discrepancy) than one generated by random sampling.
        """
        # Define specific test instance setup
        kernel = coreax.kernel.SquaredExponentialKernel()
        num_data_points = 100

        # Define some data - sufficiently large that we would not expect a random sample
        # to typically compete with a kernel herding approach
        generator = np.random.default_rng(self.random_data_generation_key)
        x = generator.random((num_data_points, self.dimension))
        data = coreax.data.ArrayData.load(x)

        test_approximator = coreax.approximation.ANNchorApproximator(
            kernel=kernel, num_kernel_points=50, num_train_points=50
        )

        # Create a kernel herding object
        coresubset_object_herding = coreax.coresubset.KernelHerding(
            kernel=kernel,
            approximator=test_approximator,
        )

        with patch.object(
            coreax.kernel.Kernel,
            "approximate_kernel_matrix_row_sum_mean",
            wraps=kernel.approximate_kernel_matrix_row_sum_mean,
        ) as mock_method:
            # Apply kernel herding on the dataset, and record the coreset for comparison
            coresubset_object_herding.fit(
                original_data=data,
                strategy=coreax.reduction.SizeReduce(self.coreset_size),
            )
            herding_coreset = coresubset_object_herding.coreset

        # Check the approximation method in the Kernel class is called exactly once
        mock_method.assert_called_once()

        # Create a random refinement object and generate a coreset, for comparison
        coresubset_object_random = coreax.coresubset.RandomSample(
            random_seed=0, unique=True
        )
        coresubset_object_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        random_coreset = coresubset_object_random.coreset

        # Define a metric and compare quality of produced coresets
        metric = coreax.metrics.MMD(kernel=kernel)
        herding_metric = metric.compute(x, herding_coreset)
        random_metric = metric.compute(x, random_coreset)
        self.assertLess(float(herding_metric), float(random_metric))

    # pylint: enable=too-many-locals
    def test_greedy_body(self) -> None:
        """
        Test the _greedy_body method of the KernelHerding class.

        Methods called by this method are mocked and assumed tested elsewhere.
        """

        with (
            patch("coreax.kernel.Kernel") as mock_kernel,
            patch("coreax.data.DataReader") as mock_reader,
            patch("coreax.validation.validate_is_instance") as _,
        ):
            # Mock some data
            mock_reader.pre_coreset_array = jnp.asarray([[0, 0], [1, 1], [2, 2]])
            # Define a kernel matrix row sum mean. On the first call of the greedy body,
            # we will select the first point in the coreset. Recall herding can be
            # thought of as a balance between selecting points in high density
            # (kernel_matrix_row_sum_mean is large) but that are not too close to points
            # already in the coreset (defined by kernel_similarity_penalty). Hence, the
            # first point selected should be the index of the largest entry in
            # kernel_matrix_row_sum_mean
            kernel_matrix_row_sum_mean = jnp.asarray([0.6, 0.75, 0.55])

            def mock_kernel_vectorised(_, __):
                """
                Evaluate a (mocked) vectorised kernel over two inputs.

                :return: Fixed valued array
                """
                return jnp.asarray([[0.5, 1, 1], [0.5, 1, 1], [0.5, 1, 1]])

            # Define class

            # Assign mock kernel after the input validation has happened, which
            # simplifies the test enormously
            test_class = coreax.coresubset.KernelHerding(kernel=mock_kernel)

            # Predefine the variables that are updated in the loop
            coreset_indices_0 = jnp.zeros(2, dtype=jnp.int32)
            kernel_similarity_penalty_0 = jnp.zeros(3)

            # Call the greedy body to get the first point in the coreset
            # Disable pylint warning for protected-access as we are testing an
            # analytically tractable part of the overall herding algorithm
            # pylint: disable=protected-access
            (coreset_indices_1, kernel_similarity_penalty_1) = test_class._greedy_body(
                i=0,
                val=(coreset_indices_0, kernel_similarity_penalty_0),
                x=mock_reader.pre_coreset_array,
                kernel_vectorised=mock_kernel_vectorised,
                kernel_matrix_row_sum_mean=kernel_matrix_row_sum_mean,
                unique=True,
            )
            # pylint: enable=protected-access

            # Index 1 has the highest value of kernel_matrix_row_sum_mean, verify this
            # was the point selected in the coreset
            np.testing.assert_array_equal(coreset_indices_1, np.asarray([1, 0]))

            # Since we have unique set to True in the greedy body call, we should have
            # set the penalty for point index 1 to be infinite
            np.testing.assert_array_equal(
                kernel_similarity_penalty_1, np.asarray([0.5, np.inf, 0.5])
            )

            # Alter the penalty applied to the points for an illustrative test. This
            # will mean that the next coreset point selected should be the data-point
            # with index 2. Recall that kernel_matrix_row_sum_mean is [0.6, 0.75, 0.55],
            # and so just from density alone, the next largest point in this is index 0.
            # However, the penalty term now makes the point with index 2 the highest
            # overall when the kernel row mean and penalties are combined. Note the
            # 2.0* here because we divide the penalty term by loop index + 1
            kernel_similarity_penalty_1 = kernel_similarity_penalty_1.at[0].set(
                2.0 * 0.59
            )

            # Call the greedy body a second time
            # Disable pylint warning for protected-access as we are testing an
            # analytically tractable part of the overall herding algorithm
            # pylint: disable=protected-access
            (coreset_indices_2, kernel_similarity_penalty_2) = test_class._greedy_body(
                i=1,
                val=(coreset_indices_1, kernel_similarity_penalty_1),
                x=mock_reader.pre_coreset_array,
                kernel_vectorised=mock_kernel_vectorised,
                kernel_matrix_row_sum_mean=kernel_matrix_row_sum_mean,
                unique=False,
            )
            # pylint: enable=protected-access

            # Index 2 should now have been added to the coreset
            np.testing.assert_array_equal(coreset_indices_2, np.asarray([1, 2]))

            # Since we have unique set to False in the greedy body call, we should not
            # have set the penalty for point index 2 to be infinite
            np.testing.assert_array_less(kernel_similarity_penalty_2[2], np.inf)


class TestRandomSample(unittest.TestCase):
    """
    Tests related to RandomSample class in coresubset.py.
    """

    def setUp(self):
        """
        Generate data for use across unit tests.

        Generate n random points in d dimensions from a uniform distribution [0, 1).

        ``n``: Number of test data points
        ``d``: Dimension of data
        ``m``: Number of points to randomly select for second dataset Y
        ``max_size``: Maximum number of points for block calculations
        """
        # Define data parameters
        self.num_points_in_data = 30
        self.dimension = 10
        self.random_data_generation_key = 0
        self.coreset_size = 10
        self.random_sampling_seed = 42

        # Define example dataset
        generator = np.random.default_rng(self.random_data_generation_key)
        x = generator.random((self.num_points_in_data, self.dimension))
        data_obj = coreax.data.ArrayData.load(x)

        self.data_obj = data_obj

    def test_tree_flatten(self) -> None:
        """
        Test that the pytree is flattened as expected.
        """
        # Create a kernel herding object
        coresubset_object_random_sample = coreax.coresubset.RandomSample(
            random_seed=self.random_sampling_seed, unique=True
        )

        # Set attributes on the object to ensure actual values are returned
        coresubset_object_random_sample.kernel_matrix_row_sum_mean = None
        coresubset_object_random_sample.coreset_indices = jnp.zeros(1, dtype=jnp.int32)
        coresubset_object_random_sample.coreset = jnp.zeros([2, 3])
        coresubset_object_random_sample.unique = False
        coresubset_object_random_sample.refine_method = "ABC"
        coresubset_object_random_sample.weights_optimiser = "DEF"
        coresubset_object_random_sample.random_seed = 1989

        # Call the method and check each output are as expected
        (
            output_children,
            output_aux_data,
        ) = coresubset_object_random_sample.tree_flatten()

        self.assertEqual(len(output_children), 4)
        self.assertIsNone(output_children[0])
        self.assertIsNone(output_children[1])
        np.testing.assert_array_equal(output_children[2], jnp.zeros(1, dtype=jnp.int32))
        np.testing.assert_array_equal(output_children[3], jnp.zeros([2, 3]))
        self.assertDictEqual(
            output_aux_data,
            {
                "unique": False,
                "refine_method": "ABC",
                "weights_optimiser": "DEF",
                "random_seed": 1989,
            },
        )

    def test_random_sample(self) -> None:
        """Test data reduction by uniform-randomly sampling a fixed number of points."""
        random_sample = coreax.coresubset.RandomSample(
            random_seed=self.random_sampling_seed, unique=True
        )
        random_sample.fit(
            original_data=self.data_obj,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )

        # Assert the number of indices in the reduced data is as expected
        self.assertEqual(len(random_sample.coreset_indices), self.coreset_size)

        # Convert lists to set of tuples
        coreset_set = set(map(tuple, np.array(random_sample.coreset)))
        orig_data_set = set(
            map(tuple, np.array(random_sample.original_data.pre_coreset_array))
        )
        # Find common rows
        num_common_rows = len(coreset_set & orig_data_set)
        # Assert all rows in the coreset are in the original dataset
        self.assertEqual(len(coreset_set), num_common_rows)

    def test_random_sample_with_replacement(self) -> None:
        """
        Test reduction of datasets by uniform random sampling with replacement.

        For the purposes of this test, the random sampling behaviour is known for the
         seeds in setUp(). The parameters self.num_points_in_coreset = 10 and
        self.random_sampling_key = 42 ensure a repeated coreset point when unique=False.
        """
        random_sample = coreax.coresubset.RandomSample(
            random_seed=self.random_sampling_seed, unique=False
        )
        random_sample.fit(
            original_data=self.data_obj,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )

        unique_reduction_indices = jnp.unique(random_sample.coreset_indices)
        self.assertTrue(
            len(unique_reduction_indices) < len(random_sample.coreset_indices)
        )


if __name__ == "__main__":
    unittest.main()
