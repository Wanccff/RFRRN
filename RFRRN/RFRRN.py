import os
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.optimize import minimize


# Read data
def process_csv_to_dict_and_excel(input_folder, output_excel=None):

    bandname=["Red","Green","Blue"]

    band_dict = {
        'Red': {},
        'Green': {},
        'Blue': {}
    }

    max_rows = 0

    for filename in os.listdir(input_folder):
        if not filename.endswith('.csv'):
            continue

        file_path = os.path.join(input_folder, filename)

        parts = filename.split('_')
        date1 = parts[3]   #IMG TIME
        date2 = parts[14]

        df = pd.read_csv(file_path)

    
        max_rows = max(max_rows, len(df))

        for band_idx in range(1, 4):
            band_name = f'{bandname[band_idx-1]}'
            img1_col = f'img1_band{band_idx}'
            img2_col = f'img2_band{band_idx}'

            img1_values = np.array(list(df[img1_col]))
            img2_values = np.array(list(df[img2_col]))

            key = (date1, date2)


            img1_values = np.asarray(img1_values)
            img2_values = np.asarray(img2_values)

            if len(img1_values) != len(img2_values):
                raise ValueError("ERROR")

            data = np.vstack([img1_values, img2_values]).T

            mean = np.mean(data, axis=0)
            cov = np.cov(data, rowvar=False)

            if np.linalg.matrix_rank(cov) < 2:
                cov += 1e-6 * np.eye(2)

            diff = data - mean
            inv_cov = np.linalg.inv(cov)
            mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

            if threshold is None:
                threshold = np.mean(mahal_dist) + 2 * np.std(mahal_dist)

            mask = mahal_dist <= threshold

            img1_values, img2_values=img1_values[mask],img2_values[mask]

            band_dict[band_name][key] = {
                'img1': img1_values, 
                'img2': img2_values,
            }

    return band_dict


# Standardized function
def standardize_banddict_zscore(banddict,epsilon=1e-6):
    all_values = []
    for key in banddict:
        for img in banddict[key]:
            all_values.extend(banddict[key][img])
    
    mean = np.mean(all_values)
    std = np.std(all_values)
    
    standardized = deepcopy(banddict)
    for key in standardized:
        for img in standardized[key]:
            standardized[key][img] = [(v - mean) / std for v in standardized[key][img]]
    min_val = min([min(standardized[key][img]) for key in standardized for img in standardized[key]])
    offset = -min_val + epsilon

    for key in standardized:
        for img in standardized[key]:
            standardized[key][img] = [v + offset for v in standardized[key][img]]
    
    return standardized, mean, std,offset


# Apply the optimized correction parameters to the original banddict data
def apply_correction_to_banddict(banddict, params_dict):
    """
    corrected_dict：{(t1, t2): {'adj_img1': adj_v1, 'adj_img2': adj_v2, },     ...}
    """
    corrected_dict = {}
    for (t1, t2), data in banddict.items():
        v1 = np.array(data['img1'])
        v2 = np.array(data['img2'])

        a1, b1 = params_dict[t1]['a'], params_dict[t1]['b']
        a2, b2 = params_dict[t2]['a'], params_dict[t2]['b']  

        adj_v1 = a1 * v1 + b1
        adj_v2 = a2 * v2 + b2

        corrected_dict[(t1, t2)] = {
            'img1': adj_v1,
            'img2': adj_v2,
        }
    
    return corrected_dict

# RFRRN
def robust_band_optimizer(input_path, delta=1.4,):
    """
    Given a dictionary banddict containing time pairs and corresponding observations,
    Fit a set of linear correction parameters (a, b) for each time point to minimize the difference of all corrected time pairs.

    Parameter
    Banddict: dictionary, in the form of {(t1, t2) : {' img1 ': v1,' img2: v2},... }
    delta: Huber Loss Threshold (used to control the impact of outliers)

    Return
    params_dict: dictionary, in the form of {t1: {'a': a1, 'b': b1},... }
    """
    banddict=process_csv_to_dict_and_excel(input_path)
    times = set()
    for (t1, t2) in banddict:
        times.add(t1)
        times.add(t2)
    times = sorted(list(times))  
    time_to_idx = {t: i for i, t in enumerate(times)}  
    n_times = len(times)  
    print(times)
    errors = []   
    pairs = []   
    
    standardized, mean, std, offset=standardize_banddict_zscore(banddict)

    for (t1, t2), data in standardized.items():
        i = time_to_idx[t1]
        j = time_to_idx[t2]
        pairs.append((i, j))
        errors.append((data['img1'], data['img2']))
    
    def huber_loss(error):
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)  
        linear = abs_error - quadratic            
        return np.sum(0.5 * quadratic**2 + delta * linear)
    
    def objective_function(params):
        a_values = params[:n_times]     
        b_values = params[n_times:]     

        total_loss = 0.0                
        count = 0                       

        for (i, j), (v1, v2) in zip(pairs, errors):
            adj_v1 = a_values[i] * np.array(v1) + b_values[i]  
            adj_v2 = a_values[j] * np.array(v2) + b_values[j]  
            diff = adj_v1 - adj_v2                   
            loss = huber_loss(diff)                  
            total_loss += loss
            count += 1

        mean_loss = total_loss / count if count else 0  

        reg_a_weight=100,reg_b_weight=20,
        reg_a = reg_a_weight* np.sum((a_values - 1.0)**2)  
        reg_b = reg_b_weight* np.sum(b_values**2)       

        if np.random.rand() < 0.01:  
            print(mean_loss,reg_a/mean_loss,reg_b/mean_loss)        
        return mean_loss + reg_a + reg_b

    def a_mean_constraint(params): 
        return np.mean(params[:n_times]) - 1.0  
    def b_mean_constraint(params):
        return np.mean(params[n_times:]) - 0.0  
    
    constraints = [
        {'type': 'eq', 'fun': a_mean_constraint},
        {'type': 'eq', 'fun': b_mean_constraint}
    ]

    initial_params = np.ones(2 * n_times)
    initial_params[:n_times] = 1  
    initial_params[n_times:] = 0  

    result = minimize(
        objective_function,
        initial_params,
        method= 'SLSQP',
        constraints=constraints,
        options={'disp': False, 'maxiter': 1000, 'ftol': 1e-6}
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    a_values = result.x[:n_times]
    b_values = result.x[n_times:]
    params_dict = {}
    
    for i, t in enumerate(times):
        params_dict[t] = {'a': a_values[i], 'b': ((a_values[i]-1)*offset+b_values[i])*std+(1-a_values[i])*mean}  # Z-score normalization with offset

    corrected_dict=apply_correction_to_banddict(banddict, params_dict)    

    params_dict_equal = {}
    a_equal=params_dict[3]['a']
    b_equal=params_dict[3]['b']
    for i, t in enumerate(times):
        params_dict_equal[t] = {'a': params_dict[t]['a']/a_equal, 'b':(params_dict[t]['b']-b_equal)/a_equal}

    return params_dict_equal,corrected_dict


if __name__ == "__main__":
    Input= "Input the difference dictionary constructed from multiple issues of images"
    params_dict,corrected_dict=robust_band_optimizer(Input, delta=1.4)