import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import jax
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mse
from scipy.optimize import minimize
from IPython.display import clear_output
import random 
from pennylane.optimize import AdamOptimizer,QNSPSAOptimizer
from utils import *
import os
from sklearn.preprocessing import normalize
from EMCost import cost__EM

import warnings
warnings.filterwarnings("ignore")



def get_state_ae_ising(n_qb_input,ae,ranges):
    @qml.qnode(qml.device('default.mixed', wires=n_qb_input, shots=1000))
    def pio(a):
        dm1 = np.matmul( np.matrix(np.conjugate(a)).T, np.matrix(a) )
        qml.QubitDensityMatrix(dm1, wires=range(n_qb_input))
        ae.get_cirq(0)
        return qml.density_matrix(ranges)
    return pio

def compare_matrix_fidelity_ising(n_qubit_autoencoder,n_trash_qubit,ae,loc=9,interval = 1):
    data=get_data(n_qubit_autoencoder)
    
    m1=[]
    for b in range(1,100,interval):
        c1=[]
        for a in range(1,100,interval):
            res1 = get_state_ae_ising(n_qubit_autoencoder,ae,list(range(n_trash_qubit-1, n_qubit_autoencoder-1,1)))([data.ground_states[a]])
            
            res2 = get_state_ae_ising(n_qubit_autoencoder,ae,list(range(n_trash_qubit-1, n_qubit_autoencoder-1,1)))([data.ground_states[b]])

            c1.append(qml.math.fidelity(res1, res2))
        m1.append(c1)
    orig=np.array([[np.abs( np.dot(data.ground_states[i], np.conjugate(data.ground_states[j])))**2 for j in range(1,100,interval)] for i in range(1,100,interval)])
    k=np.array(m1)-orig
    m1_ar=np.array(m1)
    k_norm = (m1_ar-m1_ar.min())/(m1_ar.max()-m1_ar.min()) - orig
    plt.figure(figsize=(21,5))
    ax=plt.subplot(1,3,1)
    sns.heatmap(m1_ar,ax=ax,vmin=0, vmax=1)
    ax.set_title('Reduced')
    plt.xlabel("p")
    plt.ylabel("p")
    
    ax=plt.subplot(1,3,2)

    sns.heatmap(orig,ax=ax,vmin=0, vmax=1)
    ax.set_title('Original')
    plt.xlabel("p")
    plt.ylabel("p")
    ax=plt.subplot(1,3,3)
    sns.heatmap(k,ax=ax)
    ax.set_title('Difference')
    plt.xlabel("p")
    plt.ylabel("p")


def compare_matrix_EM_ising(n_qubit_autoencoder,n_trash_qubit,ae,loc=9,interval = 1):
    list_op_support=[1,2,3]
    list_op_support_probs=[1., 1., 1.]
    list_op_support_max_range=[1, 5, 3]
    i =1
    mq=n_qubit_autoencoder-n_trash_qubit
    set_global( mq,
        mq,
        n_trash_qubit,
        list_op_support[:mq],
        list_op_support_probs[:mq],
        True,
        list_op_support_max_range[:mq],
        use_jax=False)

    data=get_data(n_qubit_autoencoder)
    m1=[]
    for b in range(1,100,interval):
        c1=[]
        for a in range(1,100,interval):
            res1 = get_state_ae_ising(n_qubit_autoencoder,ae,list(range(n_trash_qubit-1, n_qubit_autoencoder-1,1)))([data.ground_states[a]])
            res2 = get_state_ae_ising(n_qubit_autoencoder,ae,list(range(n_trash_qubit-1, n_qubit_autoencoder-1,1)))([data.ground_states[b]])
            c1.append(cost__EM([res1])([res2]))
        m1.append(c1)

    
    mq=n_qubit_autoencoder
    set_global( mq,
        mq,
        n_trash_qubit,
        list_op_support[:mq],
        list_op_support_probs[:mq],
        True,
        list_op_support_max_range[:mq],
        use_jax=False)
    orig=[]
    for b in range(1,100,interval):
        d1=[]
        for a in range(1,100,interval):
            d1.append(cost__EM([np.outer(data.ground_states[a], np.conjugate(data.ground_states[a]))])([np.outer(data.ground_states[b], np.conjugate(data.ground_states[b]))]))
        orig.append(d1)
    k=np.array(m1)-orig
    m1_ar=np.array(m1)
    k_norm = (m1_ar-m1_ar.min())/(m1_ar.max()-m1_ar.min()) - orig
    plt.figure(figsize=(21,5))
    ax=plt.subplot(1,3,1)
    sns.heatmap(m1_ar,ax=ax,vmin=0)
    ax.set_title('Reduced')
    plt.xlabel("p")
    plt.ylabel("p")
    
    ax=plt.subplot(1,3,2)

    sns.heatmap(orig,ax=ax,vmin=0)
    ax.set_title('Original')
    plt.xlabel("p")
    plt.ylabel("p")
    ax=plt.subplot(1,3,3)
    sns.heatmap(k,ax=ax)
    ax.set_title('Difference')
    plt.xlabel("p")
    plt.ylabel("p")


# # Function to compute the reduced density matrix for specified qubits
# def reduced_density_matrix(rho, target_qubits, total_qubits):
#     """Compute the reduced density matrix for specified target_qubits
#     by tracing out the other qubits in a mixed state.
    
#     Args:
#         rho (np.ndarray): The mixed state density matrix.
#         target_qubits (list): List of qubits to keep in the reduced density matrix.
#         total_qubits (int): Total number of qubits in the system.
        
#     Returns:
#         np.ndarray: The reduced density matrix for the target qubits.
#     """
#     # Determine which qubits to trace out
#     trace_out_qubits = sorted(set(range(total_qubits)) - set(target_qubits))
    
#     # Start with the original density matrix
#     new_density_matrix = rho

#     # Iteratively trace out each qubit in trace_out_qubits
#     for qubit in trace_out_qubits:
#         # Calculate the new dimension after tracing out one qubit
#         dim = 2 ** (total_qubits - 1)  # Dimension after tracing out one qubit
#         shape = new_density_matrix.shape
        
#         # Create a new density matrix for the remaining subsystem
#         reduced_dim = int(new_density_matrix.shape[0] / 2)
#         new_density_matrix = np.zeros((reduced_dim, reduced_dim), dtype=complex)
        
#         # Loop over the basis states of the qubit to be traced out
#         for i in range(2):  # Loop for |0> and |1>
#             # Calculate the indices to trace out
#             indices = [j for j in range(shape[0]) if (j >> qubit) & 1 == i]
#             # Extract the relevant block and sum it into the new density matrix
#             new_density_matrix += new_density_matrix[np.ix_(indices, indices)]

#         # Update the density matrix after tracing out the qubit
#         rho = new_density_matrix

#     # Return the reduced density matrix for the target qubits
#     return new_density_matrix
