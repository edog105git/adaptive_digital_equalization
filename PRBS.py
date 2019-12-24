import numpy as np

def PRBS(n_bit, tap, n_sequence):
    # Generates pseudo-random binary sequence using shift register
    # INPUTS:
    # --- n_bit = number of bits in shift register
    # --- tap = index of feedback tap (from 0 to n_bit-1). E.g. if n_bit = 15, good indices to tap would be 0, 3, 6, 7, 10, 13
    # --- n_sequence = length of output PRBS
    # OUTPUT:
    # --- output = pseudo-random binary sequence with length n_sequence

    output = np.zeros(n_sequence, dtype=int)
    
    # 1. Generate Seed
    register = np.random.randint(0, 2, n_bit)
    rand_ind = np.random.randint(0, n_bit) # Prevent seed from being all zeros
    register[rand_ind] = 1

    # 2. Use shift register to generate pseudo-random binary sequence
    for i in range(n_sequence):
    # 2a. output[i] is last bit of register
        output[i] = register[n_bit-1]
    # 2b. XOR tapped bit and last bit of register
        if register[tap] + register[n_bit-1] == 1:
            temp = 1
        else:
            temp = 0
    # 2c. Shift right by one position
        register[1:n_bit] = register[0:n_bit-1]
    # 2d. Index 0 of register becomes result of XOR from 2b.
        register[0] = temp
    
    return output

