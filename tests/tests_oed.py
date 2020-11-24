import numpy as np

import pytest

from pyfoomb.oed import CovOptimality


@pytest.fixture()
def cov_evaluator():
    cov_evaluator = CovOptimality()
    return cov_evaluator


class TestCovOptimality():

    @pytest.mark.parametrize(
        "criterion", 
        [
            'A',
            'D',
            'E',
            'E_mod',
            'unknown_criterion'
        ]
    )
    def test_calculate_criteria(self, criterion, cov_evaluator):
        Cov = np.random.rand(3, 3) + 0.001
        if criterion == 'unknown_criterion':
            with pytest.raises(KeyError):
                cov_evaluator.get_value(criterion, Cov)
        else:
            cov_evaluator.get_value(criterion, Cov)

    def test_bad_Cov(self, cov_evaluator):
        # Can only use a square Cov
        with pytest.raises(ValueError):
            cov_evaluator.get_value(
                criterion='A', 
                Cov=np.random.rand(3, 4) + 0.001,
            )
        # Return nan for Covs with inf entries
        bad_Cov = np.full(shape=(3, 3), fill_value=1.0)
        bad_Cov[0, 0] = np.inf
        assert np.isnan(cov_evaluator.get_value(criterion='A', Cov=bad_Cov,))
