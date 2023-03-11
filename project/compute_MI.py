# py file to compute the MI between a Layer and the Input or the Output

import numpy as np



def get_unique_probs(digits):
    '''
    Get the probability of each digit, together with the inverse transformation
    '''
    
    unique_values, unique_counts = np.unique(digits, return_index=False, return_inverse=False, return_counts=True)
    return unique_values, unique_counts / float(sum(unique_counts))


def get_entropy_layer(x_digits):
    '''
    Compute the entropy of layer T, given the layer output x already digitalized
    '''
    
    # compute the corresponding probabilities
    _, p_layer_T = get_unique_probs(x_digits)
    
    # return the entropy, the probabilities, the indices to reconstruct the original array from the unique array
    return -np.sum(p_layer_T * np.log2(p_layer_T)) # , p_layer_T, unique_inverse
    


def compute_MI_binning(data_layer, data_output=None, nbins=30, bin_max=1, bin_min=-1, wrt_input=True):
    '''
    Function to compute the MI between the layer T and the input X or the output Y
    
    Variables:
    - data_layer  --> layer output for each input; shape [n_input, 2000]
    - data_output --> output data, i.e. a vector containing for each input the predicted class, from 1 to 10; shape [n_input, 1]
    - nbins       --> number of bins
    - wrt_input   --> if True, compute MI of layer wrt the Input, if False wrt the Output
    
    '''
    
    binsize= (bin_max-bin_min)/nbins
    N_in = data_layer.shape[0]
    
    # convert continuous value to digits, get index corresponding to the bin, still shape [n_input, 2000]
    digitalize = np.floor(data_layer/binsize).astype('int')
    
    # entropy of layer T
    H_layer = get_entropy_layer(digitalize) 
    
    # entropy of T|X
    if wrt_input:
        H_given_X = 0
        
        for i in range(N_in):
            H_given_x_small = get_entropy_layer(digitalize[i, :])
            H_given_X += 1/N_in*H_given_x_small
            
        MI = H_layer - H_given_X
        
    # entropy of T|Y
    else:
        H_given_Y = 0
        
        sorted_cl, p_cl = get_unique_probs(data_output)
        
        for i, p_i  in zip(sorted_cl, p_cl):
            mask = data_output == i
            
            H_given_y_small = get_entropy_layer(digitalize[mask, :])
            H_given_Y += p_i*H_given_y_small
        
        MI = H_layer - H_given_Y
    
    # return MI
    return MI
    
    
    