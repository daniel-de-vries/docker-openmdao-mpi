from mpi4py import MPI
import numpy as np
from openmdao.api import Problem, ScipyOptimizeDriver, ExplicitComponent, IndepVarComp
import sys


class Paraboloid(ExplicitComponent):

    def initialize(self):
        self.options.declare('x0', types=np.ndarray)
        self.options.declare('r', types=np.ndarray)

    def setup(self):
        n = self.options['x0'].size

        self.add_input('x', shape=n)
        self.add_output('y', shape=1)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['y'] = 0.
        for i, x in enumerate(inputs['x']):
            outputs['y'] += (x - self.options['x0'][i]) ** 2 / self.options['r'][i] ** 2


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 2

    if rank == 0:
        x0 = 10 * (np.random.rand(n) - 0.5)
        r = np.random.rand(n)
    else:
        x0 = np.empty(n)
        r = np.empty(n)
    comm.Bcast(x0, root=0)
    comm.Bcast(r, root=0)

    ivc = IndepVarComp()
    ivc.add_output('x', val=np.zeros(n))

    prob = Problem()
    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem('F', Paraboloid(x0=x0, r=r), promotes=['*'])
    prob.model.add_design_var('x', lower=-10, upper=10)
    prob.model.add_objective('y')

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['tol'] = 1e-8

    if rank != 0:
        prob.driver.options['disp'] = False

    prob.set_solver_print(0)

    prob.setup()
    prob.run_driver()

    if rank == 0:
        print('Exact optimum is at:')
        print(x0)
        print('Optimum found at:')
        print(prob['x'])

        if np.sum((prob['x'] - x0)**2)**0.5 / n < 1e-3:
            print('Points are close')
        else:
            print('Points are not close')
