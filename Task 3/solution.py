"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Sum, RBF, WhiteKernel, RationalQuadratic
from scipy.stats import norm

# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

np.random.seed(51)

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.x=[]
        self.f=[]
        self.v=[]
        
        self.gp_objective=GaussianProcessRegressor(kernel= WhiteKernel(0.5**2)+0.2499 * Matern(length_scale=10, nu=2.5), alpha=0.14)
        self.gp_contraint= GaussianProcessRegressor(kernel=  WhiteKernel(0.0001**2)+ConstantKernel(3.9)+np.sqrt(2) * Matern(length_scale=0.5, nu=2.5), alpha=0.0002)
 #GaussianProcessRegressor(kernel=np.sqrt(1.2) * RBF(length_scale=0.)+ 0.01*ConstantKernel()+0.001*Matern(nu=0.01), alpha=4)
        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
      
        
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        return self.optimize_acquisition_function()
        raise NotImplementedError

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def calculate_ei(self,mu, sigma, f_best):
        z = (mu - f_best) / sigma
        ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei
    
    
    def calculate_eic(self,x):
        mu_obj, sigma_obj = self.gp_objective.predict(x.reshape(-1, 1), return_std=True)
    
    # Constraint model predictions
        mu_const, sigma_const =self.gp_contraint.predict(x.reshape(-1, 1), return_std=True)
    
    # Probability of satisfying constraints
        p_constraint_satisfied = norm.cdf(4, loc=mu_const, scale=sigma_const)
    
    # Expected Improvement (EI)
      #  f_best=
      #  f_best=max(self.f)
        f_x_all = self.gp_objective.predict(np.array(self.x).reshape(-1, 1))
        f_best = np.max(f_x_all)
        ei = self.calculate_ei(mu_obj, sigma_obj, f_best)
    
    # Expected Improvement under Constraints (EIC)
        eic = ei * p_constraint_satisfied
        return eic
    
    
    
    
    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        return self.calculate_eic(x)
        # TODO: Implement the acquisition function you want to optimize.
  #      raise NotImplementedError

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
            
        """
        self.x.append(x)
        self.f.append(f)
        self.v.append(v)
        
        self.gp_objective.fit(np.array(self.x).reshape(-1, 1), np.array(self.f))
        self.gp_contraint.fit(np.array(self.x).reshape(-1, 1), np.array(self.v))
        
        
        
        # TODO: Add the observed data {x, f, v} to your model.
     #   raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        indices_less_than_4 = [index for index, value in enumerate(self.v) if value < 4]
   #     f_values=self.f[indices_less_than_4]
        f_values=[self.f[i] for i in indices_less_than_4]
       # x_values=self.x[indices_less_than_4]
        x_values=[self.x[i] for i in indices_less_than_4]
        ind=np.argmax(f_values)
        return x_values[ind]
            
            
            
        raise NotImplementedError

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + 0.15*np.randn()
        cost_val = v(x) + 0.0001*np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()