#!/usr/bin/env python
import sys, os, argparse, glob

import dqs
import numpy as np
import pickle
import qiskit
from qiskit.opflow import PauliSumOp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='path to dqs project')
    parser.add_argument('--hamiltonian', type=str, default=None,
                        help='System type: {molecular, random-binary, random-continuous}')
    parser.add_argument('--hpath', type=str, default=None,
                        help='glob path to the benchmark Hamiltonian(s)')
    parser.add_argument('--nq', type=int, default=3,
                        help='Number of qubits')
    parser.add_argument('--nt', type=int, default=10,
                        help='Number of terms')
    parser.add_argument('--nh', type=int, default=5,
                        help='Number of Hamiltonians to evaluate (random only)')
    parser.add_argument('--rvalmin', type=int, default=1,
                         help='Min r')
    parser.add_argument('--rvalmax', type=int, default=5,
                         help='Max r')
    parser.add_argument('--rvalstep', type=int, default=1,
                         help='Stepsize for r')
    parser.add_argument('--savestr', type=str, default='',
                         help='Helpful label for the saved file name')
    args = parser.parse_args()
    return args


def get_pauli_sum_op(H):
    pauli_table = qiskit.quantum_info.PauliTable.from_labels([term[1] for term in H])
    sparse_pauli_op = qiskit.quantum_info.SparsePauliOp(pauli_table, [term[0] for term in H])
    return PauliSumOp(sparse_pauli_op)


def get_exact_unitary(H):
    pauli_sum_op = get_pauli_sum_op(H)
    return pauli_sum_op.exp_i().to_matrix()


def fidelity_over_r(H, sort_type, r_vals):

    exactU = get_exact_unitary(H)

    exact_H_str = 'EXACT: ' + dqs.hamiltonians.get_H_str(H)

    dynamics = dqs.quantum_dynamics.Dynamics(exact_H_str)
    dynamics.sort_hamiltonian(sort_type=sort_type, print_info=False)

    fidelities = []
    for r in r_vals:
        dynamics.gen_circuit(t=1, r=r)
        circuitU = dynamics.getCircuitUnitary()
        fidelities.append(dynamics.processFidelity(exactU, circuitU))

    return fidelities


def main():
    args = parse_args()

    DQSROOT = args.path
    if DQSROOT[-1] != '/':
        DQSROOT += '/'
    sys.path.append(DQSROOT)

    savepath = DQSROOT + f'benchmark_results/process_fidelity_simulation/{args.nq}qubits/'
    os.makedirs(savepath, exist_ok=True)

    # Gather the Hamiltonians (either molecular or random)
    hamiltonians = []
    if args.hamiltonian == 'molecular':
        hfiles = glob.glob(f'{DQSROOT}{args.hpath}')
        for hfile in hfiles:
            _, H = dqs.hamiltonians.parseHfile(hfile)
            name = hfile.split('/')[-1][:-4]
            hamiltonians.append((name, H))
    elif args.hamiltonian == 'random-binary':
        for _ in range(args.nh):
            H = dqs.hamiltonians.random_H(args.nq, args.nt, binary=True)
            name = dqs.hamiltonians.get_H_str(H, sigfigs=2)
            hamiltonians.append((name, H))
    elif args.hamiltonian == 'random-continuous':
        for _ in range(args.nh):
            H = dqs.hamiltonians.random_H(args.nq, args.nt, binary=False)
            name = dqs.hamiltonians.get_H_str(H, sigfigs=2)
            hamiltonians.append((name, H))
    else:
        raise Exception(f'Unknown Hamiltonian type: {args.hamiltonian}')

    # DQS Simulation
    all_data = []
    r_vals = np.arange(args.rvalmin, args.rvalmax, args.rvalstep)
    for name, H in hamiltonians:
        labels = ['lex', 'mc_tsp', 'mag', 'rand']
        fidelities = {}
        for label in labels:
            fidelities[label] = fidelity_over_r(H, label, r_vals)
        all_data.append((name, H, fidelities))

    savefn = f'{args.hamiltonian}_{args.savestr}.pickle'
    with open(savepath+savefn, 'wb') as pf:
        pickle.dump({'data': all_data, 'r': r_vals}, pf)


if __name__ == '__main__':
    main()
