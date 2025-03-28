import numpy as np
import pandas as pd

def toyswift(nu, r, mt=200, eta=-3, NW=10, max_saccades=20, skip_ms=10, return_activation_values=False):
    # Processing span normalization constant
    sigma = 1 / (1 + 2 * nu + nu**2)
    
    # Activation
    a = np.zeros(NW)  # Word activation
    
    # Fixation duration Gamma distribution parameters
    shape = 9
    rate = shape / mt  # Gamma density rate
    
    # Initialize variables
    time = 0  # Time
    k = 0  # Fixated word (0-based indexing)
    act = []  # Store word saliencies
    traj = np.zeros((max_saccades,4))  # Store trajectory
    # indexing for storing trajectory values
    i = 0
    while i<max_saccades:
        # 1. Generate fixation duration
        tfix = np.random.gamma(shape, 1/rate)
        traj[i,:] = [time, k + 1, tfix, 1]  # Store trajectory (1-based indexing for Fixation)
        
        # 2. Update processing rates
        lambda_ = np.zeros(NW)
        if k - 1 >= 0:
            lambda_[k - 1] = nu * sigma
        lambda_[k] = sigma
        if k + 1 < NW:
            lambda_[k + 1] = nu * sigma
        if k + 2 < NW:
            lambda_[k + 2] = nu**2 * sigma
        
        # 3. Evolve activations
        for _ in range(0, int(tfix), skip_ms):
            time += skip_ms
            a += r * lambda_ * skip_ms / 1000
            a[a > 1] = 1
            # Compute word saliencies
            s = np.sin(np.pi * a)
            s[a >= 1] = 0
            act.append([time] + s.tolist())
        
        # 4. Check for simulation end condition
        if np.all(a >= 1) or k == NW - 1:
            break
        
        # 5. Compute target selection probabilities
        if np.sum(s) == 0:
            idx = np.where(a == 0)[0]
            if len(idx) > 0:
                s[idx[0]] = 1
            else:
                s[NW - 1] = 1
        s += 10**eta
        p = s / np.sum(s)
        
        # 6. Select saccade target
        k = np.random.choice(NW, p=p)
        i += 1
    if return_activation_values:
        trajectory_df = pd.DataFrame(traj, columns=["Time", "Fixation", "Duration", "Observed"])
        activations_df = pd.DataFrame(act, columns=["Time"] + [i+1 for i in range(NW)])
        return {"trajectory": trajectory_df, "activations": activations_df}
    else:
        return traj