import numpy

class CovOptimality():
    """
    Manages evaluation of a parameter variance-covariance matrix w.r.t. different optimality measures. 
    Typically used for optimal experimental design (OED) methods.
    """

    def get_value(self, criterion:str, Cov:numpy.ndarray) -> float:
        """
        Arguments
        ---------
            criterion : str 
                Must be one of the implemented criteria, describing the mapping of the Cov matrix to a scalar.
            Cov : numpy.ndarray
                Variance-covariance matrix, must be sqaure and positive semi-definite.

        Returns
        -------
            float
                The result of the requested mapping function Cov -> scalar.

        Raises
        ------
            ValueError
                Cov is not square.
        """

        if Cov.shape[0] != Cov.shape[1]:
            raise ValueError('Parameter covariance matrix must be square')
        if numpy.isinf(Cov).any():
            return numpy.nan
        else:
            opt_fun = self._get_optimality_function(criterion)
            return opt_fun(Cov)


    def _get_optimality_function(self, criterion:str):
        """
        Selects the criterion function.
        """

        opt_functions = {
            'A' : self._A_optimality,
            'D' : self._D_optimality,
            'E' : self._E_optimality,
            'E_mod' : self._E_mod_optimality,
            }

        return opt_functions[criterion]


    def _A_optimality(self, Cov:numpy.ndarray) -> float:
        """
        The A criterion simply adds up the parameter variances, 
        neglecting parameter covariances.
        """

        return numpy.trace(Cov)


    def _D_optimality(self, Cov:numpy.ndarray) -> float:
        """
        Evaluates the hypervolume of the parameter joint confidence ellipsoid.
        """

        return numpy.linalg.det(Cov)


    def _E_optimality(self, Cov) -> float:
        """
        Evaluate the major axis of the parameter joint confidence ellipsoid.
        """
        
        return numpy.max(numpy.linalg.eigvals(Cov))


    def _E_mod_optimality(self, Cov:numpy.ndarray) -> float:
        """
        Evaluates the 'sphericalitcity' (i.e., shape) of the parameter joint confidence ellipsoid.
        Sometimes also referred to as K-criterion.
        """

        eig_vals = numpy.linalg.eigvals(Cov)
        return numpy.min(eig_vals) / numpy.max(eig_vals)