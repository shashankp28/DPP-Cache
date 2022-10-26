import math
import numpy as np
from scipy.optimize import minimize


def constrained_solve(theta, C, nu, X_t_1, delta_t, Q_t, V, files):
    
    
    def objective(X):
        cache_hit = np.dot(theta, X)
        change = np.linalg.norm(X-X_t_1, ord=1)
        queue_term = Q_t*(change - nu)*delta_t
        return queue_term - V*cache_hit

    
    def greater_than_0(X):
        return X
    
    def less_than_1(X):
        return 1-X
    
    def cache_contraint(X):
        return C - np.sum(X)
    
    def solve():
        constraints = (
            {
                'type': 'ineq',
                'fun': cache_contraint
            },
            {
                'type': 'ineq',
                'fun': greater_than_0
            },
            {
                'type': 'ineq',
                'fun': less_than_1
            }
        )
        init = np.random.uniform(0, 1, size=(X_t_1.shape[0],))
        solution = minimize(objective, init, constraints=constraints)
        return solution
    
    def filter(X):
        indices = np.argsort(X)[::-1][:C]
        final = np.zeros((files,))
        final[indices] = 1
        return final
        
        
    X_sol = solve()
    return filter(X_sol["x"]), X_sol["fun"]


def constrained_solve_ftpl(theta, X_t_1, C, gamma, files, i):
    
    
    def objective(X):
        lr = math.sqrt(i/C)
        estimate = theta + gamma*lr
        return -np.dot(X, estimate)

    
    def greater_than_0(X):
        return X
    
    def less_than_1(X):
        return 1-X
    
    def cache_contraint(X):
        return C - np.sum(X)
    
    def solve():
        constraints = (
            {
                'type': 'ineq',
                'fun': cache_contraint
            },
            {
                'type': 'ineq',
                'fun': greater_than_0
            },
            {
                'type': 'ineq',
                'fun': less_than_1
            }
        )
        init = np.random.uniform(0, 1, size=(X_t_1.shape[0],))
        solution = minimize(objective, init, constraints=constraints)
        return solution
    
    def filter(X):
        indices = np.argsort(X)[::-1][:C]
        final = np.zeros((files,))
        final[indices] = 1
        return final
        
        
    X_sol = solve()
    return filter(X_sol["x"]), X_sol["fun"]