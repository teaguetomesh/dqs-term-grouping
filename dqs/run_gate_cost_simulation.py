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
    parser.add_argument('--hpath', type=str, default=None,
                        help='glob path to the benchmark Hamiltonian(s)')
    parser.add_argument('--rlimit', type=int, default=5,
                         help='Max r')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Desired simulation accuracy')
    parser.add_argument('--savestr', type=str, default='',
                         help='Helpful label for the saved file name')
    args = parser.parse_args()
    return args


def get_exact_unitary(H, t):
    if isinstance(H[0][1], str):
        H = [(term[1], term[0]) for term in H]
    pauli_sum_op = PauliSumOp.from_list(H, coeff=t)
    return pauli_sum_op.exp_i().to_matrix()


def reach_epsilon(t, epsilon, dqs_obj, r_limit=10):
    Uexact = get_exact_unitary(dqs_obj._getH_without_groups(), t)
    cur_epsilon = 100
    r = 0
    while cur_epsilon > epsilon and r < r_limit:
        r += 1
        dqs_obj.gen_circuit(t=t, r=r)
        Uapprox = dqs_obj.getCircuitUnitary()
        cur_epsilon = dqs_obj.diamondNorm(Uexact, Uapprox)

    if r >= r_limit:
        print(f"\t\tUnable to achieve epsilon <= {epsilon} with r <= {r_limit}")
        return -1, r, -1

    cnot_cost = dqs_obj.total_cnot_count(r=r)

    return cur_epsilon, r, cnot_cost


def main():
    args = parse_args()

    DQSROOT = args.path
    if DQSROOT[-1] != '/':
        DQSROOT += '/'
    sys.path.append(DQSROOT)

    savepath = DQSROOT + f'benchmark_results/gate_cost_simulation/epsilon{args.epsilon}/'
    os.makedirs(savepath, exist_ok=True)

    # Gather the Hamiltonians
    hfiles = glob.glob(f'{DQSROOT}{args.hpath}')

    # DQS Simulation
    for i, fn in enumerate(hfiles):
        molecule = fn.split('/')[-1][:-4]
        hamiltonian = dqs.quantum_dynamics.Dynamics(fn)._getH_without_groups()
        print(f'[{i+1}/{len(hfiles)}], {molecule}')

        results = {'lex': [], 'mag': [], 'mc_tsp': [], 'rand': [], 'depletegroups': []}
        for sort_type in results.keys():
            print(f'\t{sort_type.upper()} results...')
            for t in np.arange(0, 1.1, 0.1):
                dqs_obj = dqs.quantum_dynamics.Dynamics(hamiltonian)
                dqs_obj.sort_hamiltonian(sort_type)
                simresult = reach_epsilon(t, args.epsilon, dqs_obj, r_limit=args.rlimit)
                if simresult[0] == -1:
                    break
                results[sort_type].append((t, *simresult))
        print('done')

        savefn = f'{molecule}_{args.savestr}.pickle'
        with open(savepath+savefn, 'wb') as pf:
            pickle.dump(results, pf)


if __name__ == '__main__':
    main()
