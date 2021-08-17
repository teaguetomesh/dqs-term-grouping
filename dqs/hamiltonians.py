"""Parse an input Hamiltonian text file and output a bitstring representation"""
import numpy as np

def get_num_qubits(Hfile):
    """
    Given the problem Hamiltonian, return the appropriate number of qubits
    needed to simulate its dynamics.

    This number does not include the single ancilla qubit that is added
    to the circuit.

    Parameters
    ----------
    Hfile : str
        filename where the Hamiltonian is stored

    Returns
    -------
    Number of qubits acted on by this Hamiltonian
    """
    numq = 0
    with open(Hfile, 'r') as hfile:
        for i, line in enumerate(hfile):
            # skip the header line in the file
            if i > 0:
                most_sig_bit = int(line.split()[-1][1:])
                if most_sig_bit > numq:
                    numq = most_sig_bit
    return numq + 1


def parseHfile(Hfile):
    """
    Parameters
    ----------
    Hfile : str
        path to the Hamiltonian text file

    Returns
    -------
    List[(float,str)]
        Each of the terms in the Hamiltonian is stored as a tuple holding the
        term coefficient and the Pauli string
    """
    Nq = get_num_qubits(Hfile)
    H = []
    with open(Hfile, 'r') as hfile:
        for i, line in enumerate(hfile):
            # skip the header line in the file
            if i > 0:
                splitline = line.split()
                paulis = splitline[1:]
                if 'j' in splitline[0]:
                    # this Hamiltonian has an imaginary coefficient -- skip
                    continue
                coef = float(splitline[0])
                # if the term is simply the Identity operator -- skip
                if paulis[0][0] != 'I':
                    pauli_arr = ['I'] * Nq
                    for op in paulis:
                        pauli_arr[int(op[1:])] = op[0]
                    pauli_arr = list(reversed(pauli_arr))
                    H += [(coef, ''.join(pauli_arr))]
    return Nq, H


def parseExactHstr(Hstr):
    """
    Turn the given string into the bitstring Hamiltonian format

    Parameters
    ----------
    Hstr : str
        a string with the form: 'EXACT: (c1)T1 + (c2)T2 + ...'

    Returns
    -------
    nq : int
        Take the length of the first term in H as number of qubits needed
    H : List[(float,str)]
        Each of the terms in the Hamiltonian is tored as a string entry in
        a list
    """
    H = []
    coefs_and_terms = Hstr.split()[1:]
    for val in coefs_and_terms:
        if val == '+': continue
        else:
            coef, paulistr = val.split(')')
            H += [(float(coef.strip('(')), paulistr)]
    # return the number of qubits, and the Hamiltonian
    return len(H[0][1]), H


def get_H_str(H, sigfigs=None):
    """
    Convert the given Hamiltonian into a string format.

    Parameters
    ----------
    H : List[(float, str)]
        The given Hamiltonian is represented as a list of tuples. Each tuple
        contains the term coefficient and Pauli string.
    """
    hstr = ''
    for i, term in enumerate(H):
        if i > 0:
            hstr += ' + '
        if sigfigs is None:
            hstr += f'({term[0]}){term[1]}'
        else:
            hstr += f'({term[0]:.{sigfigs}f}){term[1]}'
    return hstr


def random_H(num_qubits, num_terms, binary=True):
    H = []
    for _ in range(num_terms):
        if binary:
            coef = np.random.choice([-1, 1])
        else:
            coef = np.random.choice([-1, 1]) * np.random.uniform(0.1, 1.0)

        while True:
            pauli_str = ''
            for _ in range(num_qubits):
                pauli_str += np.random.choice(['I', 'X', 'Y', 'Z'])
            if pauli_str not in [term[1] for term in H] and pauli_str != 'I'*num_qubits:
                break

        H.append((coef, pauli_str))
    return H
