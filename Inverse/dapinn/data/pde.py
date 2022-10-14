import numpy as np

from .data import Data
from .. import backend as bkd
from .. import config
from ..backend import backend_name
from ..utils import get_num_args, run_if_all_none


class PDE(Data):
    """ODE or time-independent PDE solver.

    Args:
        geometry: Instance of ``Geometry``.
        pde: A global PDE or a list of PDEs. ``None`` if no global PDE.
        bcs: A boundary condition or a list of boundary conditions. Use ``[]`` if no
            boundary condition.
        num_domain (int): The number of training points sampled inside the domain.
        num_boundary (int): The number of training points sampled on the boundary.
        train_distribution (string): The distribution to sample training points. One of
            the following: "uniform" (equispaced grid), "pseudo" (pseudorandom), "LHS"
            (Latin hypercube sampling), "Halton" (Halton sequence), "Hammersley"
            (Hammersley sequence), or "Sobol" (Sobol sequence).
        anchors: A Numpy array of training points, in addition to the `num_domain` and
            `num_boundary` sampled points.
        exclusions: A Numpy array of points to be excluded for training.
        solution: The reference solution.
        num_test: The number of points sampled inside the domain for testing PDE loss.
            The testing points for BCs/ICs are the same set of points used for training.
            If ``None``, then the training points will be used for testing.
        auxiliary_var_function: A function that inputs `train_x` or `test_x` and outputs
            auxiliary variables.

    Warning:
        The testing points include points inside the domain and points on the boundary,
        and they may not have the same density, and thus the entire testing points may
        not be uniformly distributed. As a result, if you have a reference solution
        (`solution`) and would like to compute a metric such as

        .. code-block:: python

            Model.compile(metrics=["l2 relative error"])

        then the metric may not be very accurate. To better compute a metric, you can
        sample the points manually, and then use ``Model.predict()`` to predict the
        solution on thess points and compute the metric:

        .. code-block:: python

            x = geom.uniform_points(num, boundary=True)
            y_true = ...
            y_pred = model.predict(x)
            error= dde.metrics.l2_relative_error(y_true, y_pred)

    Attributes:
        train_x_all: A Numpy array of all points for training. `train_x_all` is
            unordered, and does not have duplication. If there is PDE, then
            `train_x_all` is used as the training points of PDE.
        train_x_bc: A Numpy array of the training points for BCs. `train_x_bc` is
            constructed from `train_x_all` at the first step of training, by default it
            won't be updated when `train_x_all` changes. To update `train_x_bc`, set it
            to `None` and call `bc_points`, and then update the loss function by
            ``model.compile()``.
        num_bcs (list): `num_bcs[i]` is the number of points for `bcs[i]`.
        train_x: A Numpy array of the points fed into the network for training.
            `train_x` is ordered from BC points (`train_x_bc`) to PDE points
            (`train_x_all`), and may have duplicate points.
        train_aux_vars: Auxiliary variables that associate with `train_x`.
        test_x: A Numpy array of the points fed into the network for testing, ordered
            from BCs to PDE. The BC points are exactly the same points in `train_x_bc`.
        test_aux_vars: Auxiliary variables that associate with `test_x`.
    """

    def __init__(
        self,
        geometry,
        pde,
        bcs,
        NN_type=0,
        num_domain=0,
        num_boundary=0,
        train_distribution="Hammersley",
        anchors=None,
        exclusions=None,
        solution=None,
        num_test=None,
        auxiliary_var_function=None,
        x_bcs = None,
    ):
        self.NN_type = NN_type
        self.geom = geometry
        self.pde = pde
        self.bcs = bcs if isinstance(bcs, (list, tuple)) else [bcs]

        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.train_distribution = train_distribution
        self.anchors = None if anchors is None else anchors.astype(config.real(np))
        self.exclusions = exclusions

        self.soln = solution
        self.num_test = num_test

        self.auxiliary_var_fn = auxiliary_var_function

        # TODO: train_x_all is used for PDE losses. It is better to add train_x_pde explicitly.
        self.train_x_all = None
        self.train_x, self.train_y = None, None
        self.train_x_bc = None
        self.num_bcs = None
        self.test_x, self.test_y = None, None
        self.train_aux_vars, self.test_aux_vars = None, None

        self.train_next_batch()
        self.test()

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
            outputs_pde = outputs
        elif backend_name == "jax":
            # JAX requires pure functions
            outputs_pde = (outputs, aux[0])

        f = []
        if self.pde is not None:
            if get_num_args(self.pde) == 2:
                f = self.pde(inputs, outputs_pde)
            elif get_num_args(self.pde) == 3:
                if self.auxiliary_var_fn is None:
                    if aux is None or len(aux) == 1:
                        raise ValueError("Auxiliary variable function not defined.")
                    f = self.pde(inputs, outputs_pde, unknowns=aux[1])
                else:
                    f = self.pde(inputs, outputs_pde, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]

        if not isinstance(loss_fn, (list, tuple)):
            loss_fn = [loss_fn] * (len(f) + len(self.bcs))
        elif len(loss_fn) != len(f) + len(self.bcs):
            raise ValueError(
                "There are {} errors, but only {} losses.".format(
                    len(f) + len(self.bcs), len(loss_fn)
                )
            )

        bcs_start = np.cumsum([0] + self.num_bcs)
        bcs_start = list(map(int, bcs_start))
        error_f = [fi[bcs_start[-1] :] for fi in f]
        losses = [
            loss_fn[i](bkd.zeros_like(error), error) for i, error in enumerate(error_f)
        ]
        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.
            error = bc.error(self.train_x, inputs, outputs, beg, end)
            losses.append(loss_fn[len(error_f) + i](bkd.zeros_like(error), error))
        return losses

    # AUG 1: train
    @run_if_all_none("train_x", "train_y", "train_aux_vars")
    def train_next_batch(self, batch_size=None):
        self.train_x_all = self.train_points()
        self.train_x = self.bc_points()
        if self.pde is not None:
            self.train_x = np.vstack((self.train_x, self.train_x_all))
        self.train_y = self.soln(self.train_x) if self.soln else None
        if self.auxiliary_var_fn is not None:
            self.train_aux_vars = self.auxiliary_var_fn(self.train_x).astype(
                config.real(np)
            )
        
        '''
        x_2 = self.train_x[:,0]*self.train_x[:,0]
        x_2 = x_2.reshape((np.size(self.train_x),1))
        self.train_x = np.hstack((self.train_x,x_2))
        '''
        
        return self.train_x, self.train_y, self.train_aux_vars

    # AUG 2: test
    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x
        else:
            self.test_x = self.test_points()
        self.test_y = self.soln(self.test_x) if self.soln else None
        if self.auxiliary_var_fn is not None:
            self.test_aux_vars = self.auxiliary_var_fn(self.test_x).astype(
                config.real(np)
            )
        
        '''
        x_2 = self.test_x[:,0]*self.test_x[:,0]
        x_2 = x_2.reshape((np.size(self.test_x),1))
        self.test_x = np.hstack((self.test_x,x_2))
        '''
        
        return self.test_x, self.test_y, self.test_aux_vars

    def resample_train_points(self):
        """Resample the training points for PDEs. The BC points will not be updated."""
        self.train_x, self.train_y, self.train_aux_vars = None, None, None
        self.train_next_batch()

    def add_anchors(self, anchors):
        """Add new points for training PDE losses. The BC points will not be updated."""
        anchors = anchors.astype(config.real(np))
        if self.anchors is None:
            self.anchors = anchors
        else:
            self.anchors = np.vstack((anchors, self.anchors))
        self.train_x_all = np.vstack((anchors, self.train_x_all))
        self.train_x = self.bc_points()
        if self.pde is not None:
            self.train_x = np.vstack((self.train_x, self.train_x_all))
        self.train_y = self.soln(self.train_x) if self.soln else None
        if self.auxiliary_var_fn is not None:
            self.train_aux_vars = self.auxiliary_var_fn(self.train_x).astype(
                config.real(np)
            )

    def replace_with_anchors(self, anchors):
        """Replace the current PDE training points with anchors. The BC points will not be changed."""
        self.anchors = anchors.astype(config.real(np))
        self.train_x_all = self.anchors
        self.train_x = self.bc_points()
        if self.pde is not None:
            self.train_x = np.vstack((self.train_x, self.train_x_all))
        self.train_y = self.soln(self.train_) if self.soln else None
        if self.auxiliary_var_fn is not None:
            self.train_aux_vars = self.auxiliary_var_fn(self.train_x).astype(
                config.real(np)
            )

    def train_points(self):
        X = np.empty((0, self.geom.dim), dtype=config.real(np))
        if self.num_domain > 0:
            if self.train_distribution == "uniform":
                X = self.geom.uniform_points(self.num_domain, boundary=False)
            else:
                X = self.geom.random_points(
                    self.num_domain, random=self.train_distribution
                )
        if self.num_boundary > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_boundary_points(self.num_boundary)
            else:
                tmp = self.geom.random_boundary_points(
                    self.num_boundary, random=self.train_distribution
                )
            X = np.vstack((tmp, X))
        if self.anchors is not None:
            X = np.vstack((self.anchors, X))
        if self.exclusions is not None:

            def is_not_excluded(x):
                return not np.any([np.allclose(x, y) for y in self.exclusions])

            X = np.array(list(filter(is_not_excluded, X)))
            X = np.vstack((tmp, X))
        
        '''
        '''
        if self.NN_type == 2:
            x_2 = X[:,0]*X[:,0]
            x_2 = x_2.reshape((np.size(X,axis=0),1))
            X= np.hstack((X,x_2))
        '''
        '''
        
        return X
    
    @run_if_all_none("train_x_bc")
    def bc_points(self):
        self.x_bcs = [bc.collocation_points(self.train_x_all) for bc in self.bcs]
        self.num_bcs = list(map(len, self.x_bcs))

        self.train_x_bc = (
            np.vstack(self.x_bcs)
            if self.x_bcs
            else np.empty([0, self.train_x_all.shape[-1]], dtype=config.real(np))
        )

        return self.train_x_bc


    def test_points(self):
        # TODO: Use different BC points from self.train_x_bc
        x = self.geom.uniform_points(self.num_test, boundary=False)
        '''
        '''
        if self.NN_type != 0:    
            x_2 = x[:,0]*x[:,0]
            x_2 = x_2.reshape((np.size(x,axis=0),1))
            x = np.hstack((x,x_2))
        '''
        '''
        x = np.vstack((self.train_x_bc, x))
        return x


class TimePDE(PDE):
    """Time-dependent PDE solver.

    Args:
        num_initial (int): The number of training points sampled on the initial
            location.
    """

    def __init__(
        self,
        geometryxtime,
        pde,
        ic_bcs,
        NN_type=0,
        num_domain=0,
        num_boundary=0,
        num_initial=0,
        train_distribution="Hammersley",
        anchors=None,
        exclusions=None,
        solution=None,
        num_test=None,
        auxiliary_var_function=None,
        x_bcs=None,
    ):
        self.num_initial = num_initial
        super().__init__(
            geometryxtime,
            pde,
            ic_bcs,
            NN_type,
            num_domain,
            num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            exclusions=exclusions,
            solution=solution,
            num_test=num_test,
            auxiliary_var_function=auxiliary_var_function,
        )

    def train_points(self):
        X = super().train_points()
        if self.num_initial >= 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_initial_points(self.num_initial)
            else:
                tmp = self.geom.random_initial_points(
                    self.num_initial, random=self.train_distribution
                )
            if self.exclusions is not None:

                def is_not_excluded(x):
                    return not np.any([np.allclose(x, y) for y in self.exclusions])

                tmp = np.array(list(filter(is_not_excluded, tmp)))
        X = np.vstack((tmp, X))
        '''
        '''
        if self.NN_type == 1:
            x_2 = X[:,0]*X[:,0]
            x_2 = x_2.reshape((np.size(X,axis=0),1))
            X= np.hstack((X,x_2))
            X[:,[1,2]]=X[:,[2,1]]
        '''
        '''
        return X
