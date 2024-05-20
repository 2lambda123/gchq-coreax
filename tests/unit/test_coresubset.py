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

import coreax.data
import coreax.kernel
import coreax.metrics
import coreax.score_matching
import coreax.util


class TestKernelHerding(unittest.TestCase):
    """
    Tests related to the KernelHerding class defined in coresubset.py.
    """

    def test_greedy_body(self) -> None:
        """
        Test the _greedy_body method of the KernelHerding class.

        Methods called by this method are mocked and assumed tested elsewhere.
        """
        with (
            patch("coreax.kernel.Kernel") as mock_kernel,
            patch("coreax.data.DataReader") as mock_reader,
        ):
            # Mock some data
            mock_reader.pre_coreset_array = jnp.asarray([[0, 0], [1, 1], [2, 2]])
            # Define a Gramian row-mean. On the first call of the greedy body,
            # we will select the first point in the coreset. Recall herding can be
            # thought of as a balance between selecting points in high density
            # (gramian_row_mean is large) but that are not too close to points
            # already in the coreset (defined by kernel_similarity_penalty). Hence, the
            # first point selected should be the index of the largest entry in
            # gramian_row_mean
            gramian_row_mean = jnp.asarray([0.6, 0.75, 0.55])

            def mock_kernel_vectorised(_, y):
                """
                Evaluate a (mocked) vectorised kernel over two inputs.

                :return: Fixed valued array
                """
                k = jnp.asarray([[0.5, 1, 1], [0.5, 1, 1], [0.5, 1, 1]])
                return k[:, y[0]]

            # Define class

            # Assign mock kernel after the input validation has happened, which
            # simplifies the test enormously
            test_class = coreax.coresubset.KernelHerding(
                self.random_key, kernel=mock_kernel
            )

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
                gramian_row_mean=gramian_row_mean,
                unique=True,
            )
            # pylint: enable=protected-access

            # Index 1 has the highest value of gramian_row_mean, verify this
            # was the point selected in the coreset
            np.testing.assert_array_equal(coreset_indices_1, np.asarray([1, 0]))

            # Since we have unique set to True in the greedy body call, we should have
            # set the penalty for point index 1 to be infinite
            np.testing.assert_array_equal(
                kernel_similarity_penalty_1,
                np.asarray([1.0, np.inf, 1.0]),
            )

            # Alter the penalty applied to the points for an illustrative test. This
            # will mean that the next coreset point selected should be the data-point
            # with index 2. Recall that gramian_row_mean is [0.6, 0.75, 0.55],
            # and so just from density alone, the next largest point in this is index 0.
            # However, the penalty term now makes the point with index 2 the highest
            # overall when the kernel row-mean and penalties are combined. Note the
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
                gramian_row_mean=gramian_row_mean,
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
        ``block_size``: Maximum number of points for block calculations
        """
        # Define data parameters
        self.num_points_in_data = 30
        self.dimension = 10
        self.random_data_generation_key = 0
        self.coreset_size = 10
        self.random_key = random.key(42)

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
            random_key=self.random_key, unique=True
        )

        # Set attributes on the object to ensure actual values are returned
        coresubset_object_random_sample.gramian_row_mean = None
        coresubset_object_random_sample.coreset_indices = jnp.zeros(1, dtype=jnp.int32)
        coresubset_object_random_sample.coreset = jnp.zeros([2, 3])
        coresubset_object_random_sample.refine_method = "ABC"
        coresubset_object_random_sample.weights_optimiser = "DEF"

        # Call the method and check each output are as expected
        (
            output_children,
            output_aux_data,
        ) = coresubset_object_random_sample.tree_flatten()

        self.assertEqual(len(output_children), 5)
        self.assertEqual(output_children[0], self.random_key)
        self.assertIsNone(output_children[1])
        self.assertIsNone(output_children[2])
        np.testing.assert_array_equal(output_children[3], jnp.zeros(1, dtype=jnp.int32))
        np.testing.assert_array_equal(output_children[4], jnp.zeros([2, 3]))

        self.assertDictEqual(
            output_aux_data,
            {
                "unique": True,
                "refine_method": "ABC",
                "weights_optimiser": "DEF",
            },
        )

    def test_random_sample(self) -> None:
        """Test data reduction by uniform-randomly sampling a fixed number of points."""
        random_sample = coreax.coresubset.RandomSample(
            random_key=self.random_key, unique=True
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
            random_key=self.random_key, unique=False
        )
        random_sample.fit(
            original_data=self.data_obj,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )

        unique_reduction_indices = jnp.unique(random_sample.coreset_indices)
        self.assertLess(
            len(unique_reduction_indices), len(random_sample.coreset_indices)
        )

    def test_random_sample_invalid_kernel(self):
        """
        Test the class RandomSample when given an invalid kernel object.
        """
        # Define a random sample object with the invalid kernel
        random_sample = coreax.coresubset.RandomSample(
            random_key=self.random_key,
            kernel=coreax.util.InvalidKernel,
            refine_method=coreax.refine.RefineRegular(),
        )

        # The fit method should just randomly select points - so we expect to be able to
        # call fit without any errors being raised
        random_sample.fit(
            original_data=self.data_obj,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )

        # Now, if we refine the coreset generated during the call to fit, we will use
        # the kernel, so expect an error to be raised
        with self.assertRaises(AttributeError) as error_raised:
            random_sample.refine()

        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'compute'",
        )

    def test_random_sample_invalid_weights_optimiser(self):
        """
        Test the class RandomSample when given an invalid weights_optimiser object.
        """
        # Define a random sample object with the invalid weights_optimiser - note that
        # InvalidKernel also does not have a solve method, so suits the purpose of
        # this test
        random_sample = coreax.coresubset.RandomSample(
            random_key=self.random_key,
            weights_optimiser=coreax.util.InvalidKernel,
        )

        # The fit method should just randomly select points - so we expect to be able to
        # call fit without any errors being raised
        random_sample.fit(
            original_data=self.data_obj,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )

        # Now, if we weight the coreset generated during the call to fit, we will use
        # the weights optimiser, so expect an error to be raised
        with self.assertRaises(AttributeError) as error_raised:
            random_sample.solve_weights()

        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'solve'",
        )

    def test_random_sample_invalid_unique(self):
        """
        Test the class RandomSample when given an invalid value for unique.
        """
        # Define a random sample object with the invalid unique parameter
        random_sample = coreax.coresubset.RandomSample(
            random_key=self.random_key,
            unique="ABC123",
        )

        # The fit method should sample with replacement if unique is set to False,
        # otherwise it will just sample without replacement, even if the input was not
        # a bool. Hence, we are just checking that all the coreset indices are unique
        random_sample.fit(
            original_data=self.data_obj,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )
        self.assertEqual(
            len(random_sample.coreset_indices.tolist()),
            len(set(random_sample.coreset_indices.tolist())),
        )

    def test_random_sample_invalid_refine_method(self):
        """
        Test the class RandomSample when given an invalid refine_method object.
        """
        # Define a random sample object with the invalid refine_method - note that
        # InvalidKernel also does not have a refine method, so suits the purpose of
        # this test
        random_sample = coreax.coresubset.RandomSample(
            random_key=self.random_key,
            refine_method=coreax.util.InvalidKernel,
        )

        # The fit method should just randomly select points - so we expect to be able to
        # call fit without any errors being raised
        random_sample.fit(
            original_data=self.data_obj,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )

        # Now, if we refine the coreset generated during the call to fit, we will use
        # the refine method, so expect an error to be raised
        with self.assertRaises(AttributeError) as error_raised:
            random_sample.refine()

        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'refine'",
        )

    def test_random_sample_fit_zero_coreset_size(self):
        """
        Test how random sample performs when given a zero value of coreset_size.
        """
        # Define a kernel herding object
        random_sample = coreax.coresubset.RandomSample(random_key=self.random_key)

        # Call the fit method with a coreset size of 0 - this should just sample zero
        # points at random, so not raise any issues
        random_sample.fit(
            original_data=self.data_obj,
            strategy=coreax.reduction.SizeReduce(coreset_size=0),
        )
        self.assertEqual(len(random_sample.coreset_indices), 0)
        self.assertEqual(len(random_sample.coreset), 0)

    def test_random_sample_fit_negative_coreset_size(self):
        """
        Test how random sample performs when given a negative value of coreset_size.
        """
        # Define a kernel herding object
        random_sample = coreax.coresubset.RandomSample(
            random_key=self.random_key,
        )

        # Call the fit method with a negative coreset size - this should try to run a
        # JAX loop with start and end points being the same, and index an empty array,
        # so raise a value error
        with self.assertRaises(ValueError) as error_raised:
            random_sample.fit(
                original_data=self.data_obj,
                strategy=coreax.reduction.SizeReduce(coreset_size=-2),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be a positive integer",
        )

    def test_random_sample_fit_float_coreset_size(self):
        """
        Test how random sample performs when given a float value of coreset_size.
        """
        # Define a kernel herding object
        random_sample = coreax.coresubset.RandomSample(
            random_key=self.random_key,
        )

        # Call the fit method with a float value for coreset size - this should error
        # when trying to define an integer number of samples
        with self.assertRaises(ValueError) as error_raised:
            random_sample.fit(
                original_data=self.data_obj,
                strategy=coreax.reduction.SizeReduce(coreset_size=2.0),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be a positive integer",
        )

    def test_random_sample_fit_invalid_size_reduce(self):
        """
        Test how random sample performs when given an invalid reduction strategy.
        """
        # Define a kernel herding object
        random_sample = coreax.coresubset.RandomSample(random_key=self.random_key)

        # Call the fit method with an invalid size reduce object, which should cause an
        # error as we have no reduce method to call
        with self.assertRaises(AttributeError) as error_raised:
            random_sample.fit(
                original_data=self.data_obj, strategy=coreax.util.InvalidKernel
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'reduce'",
        )

    def test_random_sample_fit_invalid_data(self):
        """
        Test how kernel herding performs when given an invalid data.
        """
        # Define a kernel herding object
        random_sample = coreax.coresubset.RandomSample(random_key=self.random_key)

        # Call the fit method with a list rather than a data object - this should error
        # as we don't have a pre_coreset_array attribute
        with self.assertRaises(AttributeError) as error_raised:
            random_sample.fit(
                original_data=[1, 2, 3],
                strategy=coreax.reduction.SizeReduce(coreset_size=2),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "'list' object has no attribute 'pre_coreset_array'",
        )


class TestRPCholesky(unittest.TestCase):
    """
    Tests related to the RPCholesky class defined in coresubset.py.
    """

    def setUp(self):
        """
        Generate data for use across unit tests.
        """
        # Define data parameters
        self.dimension = 3
        self.random_data_generation_key = 0
        self.coreset_size = 20
        self.random_key = random.key(0)

        # Define some generic data for use in input validation
        generator = np.random.default_rng(self.random_data_generation_key)
        self.generic_data = coreax.data.ArrayData.load(
            generator.random((3, self.dimension))
        )

    def test_tree_flatten(self) -> None:
        """
        Test that the pytree is flattened as expected.
        """
        # Create a ROC object
        kernel = coreax.kernel.SquaredExponentialKernel()
        coresubset_object_not_random = coreax.coresubset.RPCholesky(
            self.random_key,
            kernel=kernel,
        )

        # Set attributes on the object to ensure actual values are returned
        coresubset_object_not_random.gramian_row_mean = None
        coresubset_object_not_random.coreset_indices = jnp.zeros(1, dtype=jnp.int32)
        coresubset_object_not_random.coreset = jnp.zeros([2, 3])
        coresubset_object_not_random.block_size = 5
        coresubset_object_not_random.unique = False
        coresubset_object_not_random.refine_method = "ABC"
        coresubset_object_not_random.weights_optimiser = "DEF"

        # Call the method and check each output are as expected
        output_children, output_aux_data = coresubset_object_not_random.tree_flatten()

        self.assertEqual(len(output_children), 5)
        self.assertEqual(output_children[0], self.random_key)
        self.assertEqual(output_children[1], kernel)
        self.assertIsNone(output_children[2])
        np.testing.assert_array_equal(output_children[3], jnp.zeros(1, dtype=jnp.int32))
        np.testing.assert_array_equal(output_children[4], jnp.zeros([2, 3]))
        self.assertDictEqual(
            output_aux_data,
            {
                "block_size": 5,
                "unique": False,
                "refine_method": "ABC",
                "weights_optimiser": "DEF",
            },
        )

    def test_fit_comparison_to_random_and_refined(self) -> None:
        """
        Test the fit method of the RPCholesky class with a simple example.

        The test checks that a coreset generated via RPC has an improved quality
        (measured by maximum mean discrepancy) than one generated by random sampling. We
        further check that if the coreset generated by RPC is refined, the quality
        improves yet again.
        """
        # Define specific test instance setup
        kernel = coreax.kernel.SquaredExponentialKernel()
        num_data_points = 100

        # Define some data - sufficiently large that we would not expect a random sample
        # to typically compete with a RPC approach
        generator = np.random.default_rng(self.random_data_generation_key)
        x = generator.random((num_data_points, self.dimension))
        metric_x = coreax.data.Data(x)
        data = coreax.data.ArrayData.load(x)

        # Create a RPC object
        coresubset_object_not_random = coreax.coresubset.RPCholesky(
            self.random_key, kernel=kernel, refine_method=coreax.refine.RefineRegular()
        )

        # Apply RPC on the dataset, and record the coreset for comparison
        coresubset_object_not_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        fitted_coresubset = coreax.data.Data(coresubset_object_not_random.coreset)

        # Create a random refinement object and generate a coreset, for comparison
        coresubset_object_random = coreax.coresubset.RandomSample(
            random_key=self.random_key, unique=True
        )
        coresubset_object_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        random_coreset = coreax.data.Data(coresubset_object_random.coreset)

        # Define a metric and compare quality of produced coresets
        metric = coreax.metrics.MMD(kernel=kernel)
        not_random_metric = metric.compute(metric_x, fitted_coresubset)
        random_metric = metric.compute(metric_x, random_coreset)
        self.assertLess(float(not_random_metric), float(random_metric))

        # Create a coreset via RPC with refinement, to check refinement
        # improves the coreset quality
        coresubset_object_not_random.refine()
        refined_fitted_coresubset = coreax.data.Data(
            coresubset_object_not_random.coreset
        )

        # Compare quality of refined coreset to non-refined coreset
        refined_not_random_metric = metric.compute(metric_x, refined_fitted_coresubset)
        self.assertLess(float(refined_not_random_metric), float(not_random_metric))

    def test_fit_comparison_to_random_and_refined_not_unique(self) -> None:
        """
        Test the non-unique fit method of the RPCholesky class with a simple example.

        The test checks that a coreset generated via RPC has an improved quality
        (measured by maximum mean discrepancy) than one generated by random sampling. We
        further check that if the coreset generated by RPC is refined, the quality
        improves yet again.
        """
        # Define specific test instance setup
        kernel = coreax.kernel.SquaredExponentialKernel()
        num_data_points = 100

        # Define some data - sufficiently large that we would not expect a random sample
        # to typically compete with a RPC approach
        generator = np.random.default_rng(self.random_data_generation_key)
        x = generator.random((num_data_points, self.dimension))
        metric_x = coreax.data.Data(x)
        data = coreax.data.ArrayData.load(x)

        # Create a RPC object
        coresubset_object_not_random = coreax.coresubset.RPCholesky(
            self.random_key,
            kernel=kernel,
            refine_method=coreax.refine.RefineRegular(),
            unique=False,
        )

        # Apply RPC on the dataset, and record the coreset for comparison
        coresubset_object_not_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        fitted_coresubset = coreax.data.Data(coresubset_object_not_random.coreset)

        # Create a random refinement object and generate a coreset, for comparison
        coresubset_object_random = coreax.coresubset.RandomSample(
            random_key=self.random_key, unique=True
        )
        coresubset_object_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        random_coreset = coreax.data.Data(coresubset_object_random.coreset)

        # Define a metric and compare quality of produced coresets
        metric = coreax.metrics.MMD(kernel=kernel)
        not_random_metric = metric.compute(metric_x, fitted_coresubset)
        random_metric = metric.compute(metric_x, random_coreset)
        self.assertLess(float(not_random_metric), float(random_metric))

        # Create a coreset via RPC with refinement, to check refinement
        # improves the coreset quality
        coresubset_object_not_random.refine()
        refined_fitted_coresubset = coreax.data.Data(
            coresubset_object_not_random.coreset
        )

        # Compare quality of refined coreset to non-refined coreset
        refined_not_random_metric = metric.compute(metric_x, refined_fitted_coresubset)
        self.assertLess(float(refined_not_random_metric), float(not_random_metric))

    def test_rp_cholesky_invalid_weights_optimiser(self):
        """
        Test the class RPCholesky when given an invalid weights_optimiser object.
        """
        # Define a RPC object with the invalid weights_optimiser - note that
        # InvalidKernel also does not have a solve method, so suits the purpose of
        # this test
        rpc_object = coreax.coresubset.RPCholesky(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
            weights_optimiser=coreax.util.InvalidKernel,
        )

        # The fit method should not use the weights optimiser at all, so is expected to
        # run without issue
        rpc_object.fit(
            original_data=self.generic_data,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )

        # Now, if we weight the coreset generated during the call to fit, we will use
        # the weights optimiser, so expect an error to be raised
        with self.assertRaises(AttributeError) as error_raised:
            rpc_object.solve_weights()

        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'solve'",
        )

    def test_rp_cholesky_invalid_refine_method(self):
        """
        Test the class RPCholesky when given an invalid refine_method object.
        """
        # Define a RPC object with the invalid refine_method - note that
        # InvalidKernel also does not have a refine method, so suits the purpose of
        # this test
        rpc_object = coreax.coresubset.RPCholesky(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
            refine_method=coreax.util.InvalidKernel,
        )

        # The fit method should not use the refine method at all, so is expected to run
        # without issue
        rpc_object.fit(
            original_data=self.generic_data,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )

        # Now, if we refine the coreset generated during the call to fit, we will use
        # the refine method, so expect an error to be raised
        with self.assertRaises(AttributeError) as error_raised:
            rpc_object.refine()
        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'refine'",
        )

    def test_rp_cholesky_fit_zero_coreset_size(self):
        """
        Test how RPC performs when given a zero value of coreset_size.
        """
        # Define a RPC object
        rpc_object = coreax.coresubset.RPCholesky(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
        )

        # Call the fit method with a coreset size of 0 - this should try to run a JAX
        # loop with start and end points being the same, and index an empty array,
        # so raise a value error
        with self.assertRaises(ValueError) as error_raised:
            rpc_object.fit(
                original_data=self.generic_data,
                strategy=coreax.reduction.SizeReduce(coreset_size=0),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be non-zero",
        )

    def test_rp_cholesky_fit_negative_coreset_size(self):
        """
        Test how RPC performs when given a negative value of coreset_size.
        """
        # Define a RPC object
        rpc_object = coreax.coresubset.RPCholesky(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
        )

        # Call the fit method with a negative coreset size - this should try to run a
        # JAX loop with start and end points being the same, and index an empty array,
        # so raise a value error
        with self.assertRaises(ValueError) as error_raised:
            rpc_object.fit(
                original_data=self.generic_data,
                strategy=coreax.reduction.SizeReduce(coreset_size=-2),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must not be negative",
        )

    def test_rp_cholesky_fit_float_coreset_size(self):
        """
        Test how RPC performs when given a float value of coreset_size.
        """
        # Define a RPC object
        rpc_object = coreax.coresubset.RPCholesky(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
        )

        # Call the fit method with a float given for coreset size - which should error
        # when we try to create a JAX array with a non-integer size
        with self.assertRaises(ValueError) as error_raised:
            rpc_object.fit(
                original_data=self.generic_data,
                strategy=coreax.reduction.SizeReduce(coreset_size=2.0),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be a positive integer",
        )

    def test_rp_cholesky_fit_invalid_size_reduce(self):
        """
        Test how RPC performs when given an invalid reduction strategy.
        """
        # Define a RPC object
        rpc_object = coreax.coresubset.RPCholesky(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
        )

        # Call the fit method with an invalid reduction strategy, which should error as
        # there is no reduce method
        with self.assertRaises(AttributeError) as error_raised:
            rpc_object.fit(
                original_data=self.generic_data, strategy=coreax.util.InvalidKernel
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'reduce'",
        )

    def test_rp_cholesky_fit_invalid_data(self):
        """
        Test how RPC performs when given an invalid data.
        """
        # Define a RPC object
        rpc_object = coreax.coresubset.RPCholesky(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
        )

        # Call the fit method with a list rather than a data object. This should error
        # as there is no attribute pre_coreset_array
        with self.assertRaises(AttributeError) as error_raised:
            rpc_object.fit(
                original_data=[1, 2, 3],
                strategy=coreax.reduction.SizeReduce(coreset_size=2),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "'list' object has no attribute 'pre_coreset_array'",
        )


class TestSteinThinning(unittest.TestCase):
    """
    Tests related to the SteinThinning class defined in coresubset.py.
    """

    def setUp(self):
        """
        Generate data for use across unit tests.
        """
        # Define data parameters
        self.dimension = 3
        self.random_data_generation_key = 0
        self.coreset_size = 20
        self.random_key = random.key(0)

        # Define some generic data for use in input validation
        generator = np.random.default_rng(self.random_data_generation_key)
        self.generic_data = coreax.data.ArrayData.load(
            generator.random((3, self.dimension))
        )

    def test_tree_flatten(self) -> None:
        """
        Test that the pytree is flattened as expected.
        """
        # Create a ROC object
        kernel = coreax.kernel.SquaredExponentialKernel()
        coresubset_object_not_random = coreax.coresubset.SteinThinning(
            self.random_key,
            kernel=kernel,
        )

        # Set attributes on the object to ensure actual values are returned
        coresubset_object_not_random.gramian_row_mean = None
        coresubset_object_not_random.coreset_indices = jnp.zeros(1, dtype=jnp.int32)
        coresubset_object_not_random.coreset = jnp.zeros([2, 3])
        coresubset_object_not_random.block_size = 5
        coresubset_object_not_random.unique = False
        coresubset_object_not_random.refine_method = "ABC"
        coresubset_object_not_random.weights_optimiser = "DEF"
        coresubset_object_not_random.score_method = "XYZ"

        # Call the method and check each output are as expected
        output_children, output_aux_data = coresubset_object_not_random.tree_flatten()

        self.assertEqual(len(output_children), 5)
        self.assertEqual(output_children[0], self.random_key)
        self.assertEqual(output_children[1], kernel)
        self.assertIsNone(output_children[2])
        np.testing.assert_array_equal(output_children[3], jnp.zeros(1, dtype=jnp.int32))
        np.testing.assert_array_equal(output_children[4], jnp.zeros([2, 3]))
        self.assertDictEqual(
            output_aux_data,
            {
                "block_size": 5,
                "unique": False,
                "refine_method": "ABC",
                "regularise": True,
                "weights_optimiser": "DEF",
                "score_method": "XYZ",
            },
        )

    def test_fit_comparison_to_random_and_refined(self) -> None:
        """
        Test the fit method of the SteinThinning class with a simple example.

        The test checks that a coreset generated via SteinThinning has an improved
        quality (measured by maximum mean discrepancy) than one generated by random
        sampling. We further check that if the coreset generated by SteinThinning is
        refined, the quality improves yet again.
        """
        num_data_points = 100

        # Define some data - sufficiently large that we would not expect a random sample
        # to typically compete with a SteinThinning approach
        generator = np.random.default_rng(self.random_data_generation_key)
        x = generator.random((num_data_points, self.dimension))
        metric_x = coreax.data.Data(x)
        data = coreax.data.ArrayData.load(x)

        # Define specific test instance setup
        length_scale = np.sqrt(num_data_points)
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Create a SteinThinning object
        coresubset_object_not_random = coreax.coresubset.SteinThinning(
            self.random_key,
            kernel=kernel,
            refine_method=coreax.refine.RefineRegular(),
            unique=True,
        )

        # Apply SteinThinning on the dataset, and record the coreset for comparison
        coresubset_object_not_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        fitted_coresubset = coreax.data.Data(coresubset_object_not_random.coreset)

        # Create a random refinement object and generate a coreset, for comparison
        coresubset_object_random = coreax.coresubset.RandomSample(
            random_key=self.random_key, unique=True
        )
        coresubset_object_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        random_coreset = coreax.data.Data(coresubset_object_random.coreset)

        # Define a metric and compare quality of produced coresets
        metric = coreax.metrics.MMD(kernel=kernel)
        not_random_metric = metric.compute(metric_x, fitted_coresubset)
        random_metric = metric.compute(metric_x, random_coreset)
        self.assertLess(float(not_random_metric), float(random_metric))

        # Create a coreset via SteinThinning with refinement, to check refinement
        # improves the coreset quality
        coresubset_object_not_random.refine()
        refined_fitted_coresubset = coreax.data.Data(
            coresubset_object_not_random.coreset
        )

        # Compare quality of refined coreset to non-refined coreset
        refined_not_random_metric = metric.compute(metric_x, refined_fitted_coresubset)
        self.assertLess(float(refined_not_random_metric), float(not_random_metric))

    def test_fit_comparison_to_random_and_refined_not_unique(self) -> None:
        """
        Test the non-unique fit method of the SteinThinning class with a simple example.

        The test checks that a coreset generated via SteinThinning has an improved
        quality (measured by maximum mean discrepancy) than one generated by random
        sampling. We further check that if the coreset generated by SteinThinning is
        refined, the quality improves yet again.
        """
        # Define specific test instance setup
        num_data_points = 100

        # Define some data - sufficiently large that we would not expect a random sample
        # to typically compete with a SteinThinning approach
        generator = np.random.default_rng(self.random_data_generation_key)
        x = generator.random((num_data_points, self.dimension))
        metric_x = coreax.data.Data(x)
        data = coreax.data.ArrayData.load(x)
        num_samples_length_scale = min(num_data_points, 1_000)
        idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
        length_scale = float(coreax.kernel.median_heuristic(x[idx]))

        # Define specific test instance setup
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Create a SteinThinning object
        coresubset_object_not_random = coreax.coresubset.SteinThinning(
            self.random_key,
            kernel=kernel,
            refine_method=coreax.refine.RefineRegular(),
            unique=False,
        )

        # Apply SteinThinning on the dataset, and record the coreset for comparison
        coresubset_object_not_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        fitted_coresubset = coreax.data.Data(coresubset_object_not_random.coreset)

        # Create a random refinement object and generate a coreset, for comparison
        coresubset_object_random = coreax.coresubset.RandomSample(
            random_key=self.random_key, unique=False
        )
        coresubset_object_random.fit(
            original_data=data, strategy=coreax.reduction.SizeReduce(self.coreset_size)
        )
        random_coreset = coreax.data.Data(coresubset_object_random.coreset)

        # Define a metric and compare quality of produced coresets
        metric = coreax.metrics.MMD(kernel=kernel)
        not_random_metric = metric.compute(metric_x, fitted_coresubset)
        random_metric = metric.compute(metric_x, random_coreset)
        self.assertLess(float(not_random_metric), float(random_metric))

        # Create a coreset via SteinThinning with refinement, to check refinement
        # improves the coreset quality
        coresubset_object_not_random.refine()
        refined_fitted_coresubset = coreax.data.Data(
            coresubset_object_not_random.coreset
        )

        # Compare quality of refined coreset to non-refined coreset
        refined_not_random_metric = metric.compute(metric_x, refined_fitted_coresubset)
        self.assertLess(float(refined_not_random_metric), float(not_random_metric))

    def test_stein_thinning_invalid_weights_optimiser(self):
        """
        Test the class SteinThinning when given an invalid weights_optimiser object.
        """
        # Define a SteinThinning object with the invalid weights_optimiser - note that
        # InvalidKernel also does not have a solve method, so suits the purpose of
        # this test
        stein_object = coreax.coresubset.SteinThinning(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
            weights_optimiser=coreax.util.InvalidKernel,
        )

        # The fit method should not use the weights optimiser at all, so is expected to
        # run without issue
        stein_object.fit(
            original_data=self.generic_data,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )

        # Now, if we weight the coreset generated during the call to fit, we will use
        # the weights optimiser, so expect an error to be raised
        with self.assertRaises(AttributeError) as error_raised:
            stein_object.solve_weights()

        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'solve'",
        )

    def test_stein_thinning_invalid_refine_method(self):
        """
        Test the class SteinThinning when given an invalid refine_method object.
        """
        # Define a SteinThinning object with the invalid refine_method - note that
        # InvalidKernel also does not have a refine method, so suits the purpose of
        # this test
        stein_object = coreax.coresubset.SteinThinning(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
            refine_method=coreax.util.InvalidKernel,
        )

        # The fit method should not use the refine method at all, so is expected to run
        # without issue
        stein_object.fit(
            original_data=self.generic_data,
            strategy=coreax.reduction.SizeReduce(self.coreset_size),
        )

        # Now, if we refine the coreset generated during the call to fit, we will use
        # the refine method, so expect an error to be raised
        with self.assertRaises(AttributeError) as error_raised:
            stein_object.refine()
        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'refine'",
        )

    def test_stein_thinning_fit_zero_coreset_size(self):
        """
        Test how SteinThinning performs when given a zero value of coreset_size.
        """
        # Define a SteinThinning object
        stein_object = coreax.coresubset.SteinThinning(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
        )

        # Call the fit method with a coreset size of 0 - this should try to run a JAX
        # loop with start and end points being the same, and index an empty array,
        # so raise a value error
        with self.assertRaises(ValueError) as error_raised:
            stein_object.fit(
                original_data=self.generic_data,
                strategy=coreax.reduction.SizeReduce(coreset_size=0),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be a positive integer",
        )

    def test_stein_thinning_fit_negative_coreset_size(self):
        """
        Test how SteinThinning performs when given a negative value of coreset_size.
        """
        # Define a SteinThinning object
        stein_object = coreax.coresubset.SteinThinning(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
        )

        # Call the fit method with a negative coreset size - this should try to run a
        # JAX loop with start and end points being the same, and index an empty array,
        # so raise a value error
        with self.assertRaises(ValueError) as error_raised:
            stein_object.fit(
                original_data=self.generic_data,
                strategy=coreax.reduction.SizeReduce(coreset_size=-2),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be a positive integer",
        )

    def test_stein_thinning_fit_float_coreset_size(self):
        """
        Test how SteinThinning performs when given a float value of coreset_size.
        """
        # Define a SteinThinning object
        stein_object = coreax.coresubset.SteinThinning(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
        )

        # Call the fit method with a float given for coreset size - which should error
        # when we try to create a JAX array with a non-integer size
        with self.assertRaises(ValueError) as error_raised:
            stein_object.fit(
                original_data=self.generic_data,
                strategy=coreax.reduction.SizeReduce(coreset_size=2.0),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be a positive integer",
        )

    def test_stein_thinning_fit_invalid_size_reduce(self):
        """
        Test how SteinThinning performs when given an invalid reduction strategy.
        """
        # Define a SteinThinning object
        stein_object = coreax.coresubset.SteinThinning(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
        )

        # Call the fit method with an invalid reduction strategy, which should error as
        # there is no reduce method
        with self.assertRaises(AttributeError) as error_raised:
            stein_object.fit(
                original_data=self.generic_data, strategy=coreax.util.InvalidKernel
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'reduce'",
        )

    def test_stein_thinning_fit_invalid_data(self):
        """
        Test how SteinThinning performs when given an invalid data.
        """
        # Define a SteinThinning object
        stein_object = coreax.coresubset.SteinThinning(
            random_key=self.random_key,
            kernel=coreax.kernel.SquaredExponentialKernel(),
        )

        # Call the fit method with a list rather than a data object. This should error
        # as there is no attribute pre_coreset_array
        with self.assertRaises(AttributeError) as error_raised:
            stein_object.fit(
                original_data=[1, 2, 3],
                strategy=coreax.reduction.SizeReduce(coreset_size=2),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "'list' object has no attribute 'pre_coreset_array'",
        )


# pylint: enable=too-many-public-methods


if __name__ == "__main__":
    unittest.main()
