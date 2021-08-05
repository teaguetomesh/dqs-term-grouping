from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.quantum_info import Operator, process_fidelity, SuperOp, Chi, Choi
import sys, math, random, queue
import numpy as np
import scipy.linalg as linalg
from copy import deepcopy
from utils import parse_hamiltonian_file as phf
from utils import permutation_heuristic
from term_grouping import findMinCliqueCover, sort_all_tsp
from transpilation import apply_transpiler


class Dynamics:
    """
    Class to implement the simulation of quantum dynamics as described
    in

    -Section 4.7 of Nielsen & Chuang (Quantum computation and quantum
     information (10th anniv. version), 2010.)

    -Whitfield et al. (https://arxiv.org/abs/1001.3855)

    -Raeisi, Wiebe, Sanders (https://arxiv.org/abs/1108.4318)

    A circuit implementing the quantum simulation can be generated for a given
    problem Hamiltonian parameterized by calling the gen_circuit() method.

    Attributes
    ----------
    H : List[(float,str)]
        A list containing the terms of the Hamiltonian whose dynamics we want
        to simulate. Hfile is the name of the file storing this Hamiltonian
    nq : int
        Number of qubits required given H
    sort_type : str
        Specifies how sortedH is currently sorted
    sortedH : List[(float,str)]
        Sorted list containing the terms of the Hamiltonian
    H_withoutCo : List[str]
        Sorted list containing only the pauli strings of the Hamiltonian
    iterate_circ_copy : QuantumCircuit
        Qiskit QuantumCircuit of a single iteration of the Trotter-Suzuki (TS)
        decomposition
    full_circ_copy : QuantumCircuit
        Qiskit QuantumCircuit of the entire TS-decomposition
    deltaTs : (r, List[float])
        Tuple containing the current r value used to generate the DQS circuit
        and a list of rotation angles (deltaT) for each Rz rotation in the
        full TS-decomposition circuit
    """
    def __init__(self, Hfile):

        if 'EXACT:' in Hfile:
            # as a useful debugging option, the Hamiltonian can be manually
            # specified. When initializing the Dynamics object, be sure to
            # strictly follow the format:
            #
            #     Hfile = 'EXACT: (coef1)term1 + (coef2)term2 + ...
            #
            # where
            #     coefs : int or float
            #     terms : IIIZZZ, XIXIZY, etc...
            self.nq, self.H = phf.parseExactHstr(Hfile)
        else:
            # parse the Hamiltonian file
            self.nq, self.H = phf.parseHfile(Hfile)

        #print('Total of {} terms in Hamiltonian'.format(len(self.H)))
        # print(self.H)

        # initialize self.sortedH
        self.sort_type = None
        self.sortedH = None
        self.H_withoutCo = None
        self.sort_hamiltonian('GIVEN')
        # initialize remaining attributes to None
        self.iterate_circ_copy = None
        self.full_circ_copy = None
        self.deltaTs = None


    def _getH_without_coef(self):
        """
        parse the Hamiltonian without the coefficients in the front
        """
        H = []
        for group in self.sortedH:
            for term in group:
                H += [term[1]]
        return H


    def _get_H_bitstr(self):
        """
        Produce a bitstring representation of the Hamiltonian where

        H = [(a_j, S_j); j = 1,...,m]

        with

        a_j = the coefficient of term j
        S_j = (S_Xj, S_Yj, S_Zj) a vector giving the positions of the Pauli
              matrices within term j
        """

        H_bitstr = []

        for group in self.sortedH:
            for term in group:
                a_j, paulistr = term
                S_Xj = [i for i, pauli in enumerate(paulistr[::-1]) if pauli == 'X']
                S_Yj = [i for i, pauli in enumerate(paulistr[::-1]) if pauli == 'Y']
                S_Zj = [i for i, pauli in enumerate(paulistr[::-1]) if pauli == 'Z']
                S_j = (S_Xj, S_Yj, S_Zj)
                H_bitstr += [(float(a_j), S_j)]

        return H_bitstr


    def compare_H_reps(self):
        for i, (H_regrep, H_bstr) in enumerate(zip(self.H, H_bitstr)):
            print('TERM {}: {} <--> {}'.format(i, H_regrep, H_bstr))


    def _sp(self, p):
        """
        Compute the function given as Eqn. 13 in Raeisi, Wiebe, Sanders.
        """
        return 1 / (4 - 4**(1 / (2*p - 1)))


    def _RWS_trotterSuzuki(self, H, p, dt):
        """
        Define a function to recursively construct a bitstring representation of
        a Trotter-Suzuki iterate with the given order and timestep

        Parameters
        ----------
        H : bitstring representation of Hamiltonian
        p : int
            the order of the Trotter-Suzuki decomposition
        dt : float
            the timestep argument passed to the Trotter-Suzuki formula

        Returns
        -------
        BASE CASE:
            U_1(dt) = exp(T_1)exp(T_2)...exp(T_m)exp(T_m)exp(T_m-1)...exp(T_1)
                    = (a_1, l_1, S_1, t_1)...(a_k, l_k, S_k, t_k)
        RECURSIVE CASE:
            U_p(dt) = [U_(p-1)(sp*dt)]^2 * U_(p-1)((1-4sp)*dt) * [U_(p-1)(sp*dt)]^2
        """

        if p == 1: # BASE CASE
            ts_bitstr = [(*hterm, dt/2) for hterm in H]
            return ts_bitstr + list(reversed(ts_bitstr))
        else: # RECURSIVE CASE
            U_end = self._RWS_trotterSuzuki(H, p-1, self._sp(p) * dt)
            U_mid = self._RWS_trotterSuzuki(H, p-1, (1 - 4 * self._sp(p)) * dt)
            return 2 * U_end + U_mid + 2 * U_end


    def _MikeAndIke(self, H, p, dt):
        """
        Construct a bitstring representation of a Trotter-Suzuki iterate using
        the decomposition given in Eqn 4.98 in Nielsen & Chuang (2010)

        Parameters
        ----------
        H : bitstring representation of Hamiltonian
        p : int
            the order of the Trotter-Suzuki decomposition
        dt : float
            the timestep argument passed to the Trotter-Suzuki formula

        Returns
        -------
        bitstring representation of a single TS iterate
        """
        return [(*hterm, dt) for hterm in H]


    def _compute_to_Z_basis(self, qreg, S_j):
        """
        Transform all qubits to the Z basis

        Parameters
        ----------
        qreg : QuantumRegister
            Register of qubits
        S_j : ([int],[int],[int])
            Tuple of arrays giving the locations of the X, Y, and Z pauli
            operations respectively

        Returns
        -------
        circ : QuantumCircuit
            Qiskit QuantumCircuit object implementing the correct change of
            basis operators for the given S_J
        """

        circ = QuantumCircuit(qreg)
        Xlocs, Ylocs, Zlocs = S_j

        for loc in Xlocs:
            circ.h(qreg[loc])
        for loc in Ylocs:
            circ.h(qreg[loc])
            circ.s(qreg[loc])

        return circ


    def _uncompute_to_Z_basis(self, qreg, S_j):
        """
        Transform all qubits back to original basis

        Parameters
        ----------
        qreg : QuantumRegister
            Register of qubits
        S_j : ([int],[int],[int])
            Tuple of arrays giving the locations of the X, Y, and Z pauli
            operations respectively

        Returns
        -------
        circ : QuantumCircuit
            Qiskit QuantumCircuit object implementing the correct change of
            basis operators for the given S_J
        """

        circ = QuantumCircuit(qreg)
        Xlocs, Ylocs, Zlocs = S_j

        for loc in Xlocs:
            circ.h(qreg[loc])
        for loc in Ylocs:
            circ.sdg(qreg[loc])
            circ.h(qreg[loc])

        return circ


    def _apply_phase_shift(self, qreg, delta_t, S_j, mode):
        """
        Simulate the evolution of exp(-i(dt)Z)

        Parameters
        ----------
        qreg : QuantumRegister
            Register of qubits
        delta_t : float
            the length of the next time step in the Hamiltonian evolution
        S_j : ([int],[int],[int])
            Tuple of arrays giving the locations of the X, Y, and Z pauli
            operations respectively
        mode : str
            Indicates which DQS implementation to use: star or ladder
        """
        circ = QuantumCircuit(qreg)
        all_locs = []
        for pauli_loc in S_j:
            for loc in pauli_loc:
                all_locs.append(loc)
        all_locs = sorted(all_locs)
        # lsb is the Least Significant quBit in this term that is acted on with
        # a non-Identity operator
        lsb = all_locs[-1]

        # apply CNOT ladder -> compute parity
        if len(all_locs) != 1:
            # if len(all_locs) == 1 then there is only a single non-Identity
            # operator in this term so no CNOT gates are needed
            if mode == 'star':
                for loc in all_locs:
                    if loc != lsb:
                        circ.cx(qreg[loc], qreg[lsb])
            elif mode == 'ladder':
                for i in range(len(all_locs)-1):
                    ctrl = all_locs[i]
                    trgt = all_locs[i+1]
                    circ.cx(qreg[ctrl], qreg[trgt])

        # apply phase shift to the Least Significant quBit (LSB)
        # RZ applies the unitary gate specified in utils/RZdef.py:
        #     RZ(phi) = exp(-i*phi*Z/2)
        #circ.RZ(2*delta_t, qreg[lsb])
        # rz applies the Qiskit Z-rotation
        circ.rz(2*delta_t, qreg[lsb])

        # append the phase shift performed by the rz gate
        self.deltaTs[1].append(2*delta_t)

        # apply CNOT ladder -> uncompute parity
        if len(all_locs) != 1:
            if mode == 'star':
                for loc in reversed(all_locs):
                    if loc != lsb:
                        circ.cx(qreg[loc], qreg[lsb])
            elif mode == 'ladder':
                for i in reversed(range(len(all_locs)-1)):
                    ctrl = all_locs[i]
                    trgt = all_locs[i+1]
                    circ.cx(qreg[ctrl], qreg[trgt])

        return circ


    def _construct_iterate_circuit(self, qreg, ts_bitstr, barriers, mode):
        """
        Create a circuit which implements a single iteration of Trotter-Suzuki

        Parameters
        ----------
        qreg : QuantumRegister
            Register of qubits
        ts_bitstr : List[(a_j, S_j, t_j)]
            A bitstring representation of the Trotter-Suzuki iterate
        barriers : bool
            Include barriers within the circuit
        mode : str
            Indicate which DQS implementation to use: star or ladder

        Returns
        -------
        iter_circ : QuantumCircuit
            A Qiskit QuantumCircuit object corresponding to the given TS bitstr
        """
        iter_circ = QuantumCircuit(qreg)

        for (a_j, S_j, t_j) in ts_bitstr:
            iter_circ += self._compute_to_Z_basis(qreg, S_j)
            iter_circ += self._apply_phase_shift(qreg, a_j*t_j, S_j, mode)
            iter_circ += self._uncompute_to_Z_basis(qreg, S_j)
            if barriers:
                iter_circ.barrier()

        return iter_circ


    def get_num_groups(self):
        """
        Return the number of commuting groups the Hamiltonian has been sorted
        into.
        """
        return len(self.sortedH)


    def gen_circuit(self, t, r=1, chi=0, func='MikeAndIke', error=0.1,
                    mode='star', barriers=True, regname=None, measure=False,
                    transpile=None, device='transmon'):
        """
        Create a circuit implementing the quantum dynamics simulation

        Parameters
        ----------
        t : float
            time of evolution
        r : int
            number of timesteps to break t into. Default value of 0 will compute
            the optimal value
        chi : int
            the order of the Trotter-Suzuki approximation. Default value of 0
            will compute the optimal value
        func : str
            Select the function used to compute the Trotter-Suzuki
            decomposition, options include: MikeAndIke, RWS_trotterSuzuki
        error : float
            the accuracy to within which we wish to approximate the unitary
            evolution matrix (2-norm), only used if r or chi are = 0
        mode : str
            specify the dynamics implementation to use, options are
            [star, ladder]
        barriers : bool
            should barriers be included in the generated circuit
        regname : str
            give a name to the QuantumRegister
        measure : bool
            should measurements be added to the end of the circuit
        transpile : int
            Set the transpiler optimization level. Setting to None will do no
            transpilation, 0 only unrolls the circuit, 1, 2, and 3 will unroll
            the circuit and apply increasingly aggressive optimizations.
        device : str
            Select the backend compilation target for this circuit. Options
            include: ['transmon', 'trappedion']

        Returns
        -------
        QuantumCircuit
            QuantumCircuit object with width = nq
        """
        # print out the current configuration of the Hamiltonian
        print('Hamiltonian currently sorted by:', self.sort_type)
        #print('H =', self.H_withoutCo)

        # Check that a correct mode is set
        mode = mode.lower()
        if mode not in ['star', 'ladder']:
            raise Exception('Unknown DQS mode: {}'.format(mode))

        # convert the sorted Hamiltonian into a bitstring representation
        H_bitstr = self._get_H_bitstr()
        #print('H bitstring representation:\n', H_bitstr)
        #self.compare_H_reps()

        # create a QuantumCircuit object
        if regname is None:
            qr = QuantumRegister(self.nq)
        else:
            qr = QuantumRegister(self.nq, name=regname)
        circ = QuantumCircuit(qr)

        m = self.get_num_groups()
        max_coef_val = max([abs(term[0]) for term in self.H])
        if chi == 0:
            # Compute the optimal value for chi as discussed in Raeisi, Wiebe,
            # Sanders (2012)
            chi = math.ceil(math.sqrt(math.log(m * max_coef_val * t / error, 25/3) / 2))

        if r == 0:
            # Compute the optimal value for r as discussed in Raeisi, Wiebe,
            # Sanders (2012)
            numerator = (2*m * (5/3)**(chi-1) * chi * max_coef_val * t)**(1+1/(2*chi))
            denominator = (error/2)**(1/(2*chi))
            r = math.ceil(numerator / denominator)

        print('Trotter-Suzuki order, chi = {}'.format(chi))
        print('Time, t = {}, repetitions, r = {}'.format(t, r))

        if func == 'MikeAndIke':
            trotter_suzuki_function = self._MikeAndIke
        elif func == 'RWS_trotterSuzuki':
            trotter_suzuki_function = self._RWS_trotterSuzuki
        else:
            raise Exception('Unknown trotter_suzuki_function: {}'.format(func))
        print('Using TS function: ', func)

        iterate_bitstr = trotter_suzuki_function(H_bitstr, chi, t/r)
        print('Produced Trotter-Suzuki decomposition: order={}, len={}'.format(
                                               chi, len(iterate_bitstr)))
        #print(iterate_bitstr)

        # If we plan on using the star+ancilla optimization, we must insert
        # barriers during generation, they will be removed after optimization
        if transpile != None and transpile > 0 and mode == 'star':
            place_barriers = True
        else:
            place_barriers = barriers
        self.deltaTs = (r,[])
        iterate_circuit = self._construct_iterate_circuit(qr, iterate_bitstr,
                                                          place_barriers, mode)
        #iterate_image = iterate_circuit.draw(output='text', line_length=200)
        #print('Copy of single iteration:\n',iterate_image)
        self.iterate_circ_copy = iterate_circuit

        for n in range(r):
            circ += iterate_circuit

        if transpile == None:
            # set the current class circuit equal to the generated circ
            # This is the circuit that getCircuitUnitary will use
            self.full_circ_copy = circ
            self.full_circ_copy_clean = circ
            return circ
        else:
            # use the transpiler to perform gate cancellations
            self.full_circ_copy_clean = deepcopy(circ)
            circ_with_transpiler = apply_transpiler(circ, transpile, mode,
                                                    barriers, device=device)
            self.full_circ_copy = circ_with_transpiler
            return circ_with_transpiler


    def getIterateCircuit(self):
        """
        Return the circuit for a single iteration of the TS-decomposition
        """
        return self.iterate_circ_copy


    def sort_hamiltonian(self, sort_type='LEXICOGRAPHIC',
                         coverFunc='rlf', print_info=True,
                         random_permutation=False, mode='star'):
        """
        Arrange the Hamiltonian terms according to some sorting algorithm.

        Parameters
        ----------
        sort_type : str
            Select the sorting function to apply to the Hamiltonian
        coverFunc : str
            Select the function to find the MIN-CLIQUE-COVER
            Options are: boppana, bronk
        print_info : bool
            print to std_out when performing TSP
        random_permutation : bool
            For the max_commute orderings only. If true, randomly permute
            the order of the cliques returned by FindMinCliqueCover,
            otherwise, use the permutation heuristic to order the cliques

        Returns
        -------
        None -> modifies the class variable self.sortedH
        """
        rng = np.random.default_rng()
        if len(self.H) == 1:
            sort_type = 'GIVEN'
        sort_type = sort_type.lower()

        if sort_type in ['lexicographic', 'lex', 'lexico']:
            # Sort into lexicographic order
            self.sortedH = [[h] for h in sorted(self.H, key=lambda x: x[1])]
        elif sort_type in ['all_tsp', 'full_tsp']:
            # Sort entire Hamiltonian via TSP heuristic
            all_tsp_H = sort_all_tsp(self.H, mode, print_info=print_info)
            self.sortedH = [[h] for h in all_tsp_H]
        elif sort_type in ['max_commute', 'mc']:
            self.sortedH = findMinCliqueCover(
                        rng.permutation(self.H), self.nq, 'GC', coverFunc, print_info=print_info)
        elif sort_type in ['max_commute_w_tsp', 'mc_w_tsp', 'mc_tsp', 'max_commute_tsp']:
            max_commute_H = findMinCliqueCover(
                        rng.permutation(self.H), self.nq, 'GC', coverFunc,
                        gate_cancellation=True, print_info=print_info, mode=mode)
            if random_permutation:
                self.sortedH = list(np.random.permutation(max_commute_H))
            else:
                # use the permutation heuristic
                self.sortedH = permutation_heuristic.gen_heuristic(max_commute_H)
        elif sort_type in ['max_commute_w_random', 'mc_w_rand', 'mc_rand']:
            max_commute_H = findMinCliqueCover(
                        rng.permutation(self.H), self.nq, 'GC', coverFunc, print_info=print_info)
            max_commute_random_H = []
            for group in max_commute_H:
                random.shuffle(group)
                max_commute_random_H.append(group)
            self.sortedH = max_commute_random_H
        elif sort_type in ['max_commute_lex', 'mc_w_lex', 'mc_lex']:
            max_commute_H = findMinCliqueCover(
                        rng.permutation(self.H), self.nq, 'GC', coverFunc,
                        print_info=print_info, mode=mode)
            self.sortedH = [sorted(group, key=lambda x: x[1]) for group in max_commute_H]
        elif sort_type in ['max_commute_w_mag', 'mc_w_mag', 'mc_mag']:
            max_commute_H = findMinCliqueCover(
                        rng.permutation(self.H), self.nq, 'GC', coverFunc, print_info=print_info)
            self.sortedH = [sorted(group, key=lambda x: abs(x[0]), reverse=True) for group in max_commute_H]
        elif sort_type in ['random', 'rand']:
            self.sortedH = [[h] for h in self.H]
            random.shuffle(self.sortedH)
        elif sort_type in ['magnitude', 'mag']:
            self.sortedH = [[h] for h in sorted(self.H, key=lambda x:abs(x[0]), reverse=True)]
        elif sort_type in ['depletegroups', 'deplete_groups', 'dg']:
            # implement the depleteGroups strategy from Tranter, et. al. 2019
            # Link: https://www.mdpi.com/1099-4300/21/12/1218
            max_commute_H = findMinCliqueCover(
                        rng.permutation(self.H), self.nq, 'GC', coverFunc, print_info=print_info)
            max_commute_H = sorted(max_commute_H, key=lambda x:len(x), reverse=True)
            clique_queue = queue.Queue(maxsize=len(max_commute_H))
            for clique in max_commute_H:
                clique = sorted(clique, key=lambda x:abs(float(x[0])))
                stack = queue.LifoQueue(maxsize=len(clique))
                for term in clique:
                    stack.put(term)
                clique_queue.put(stack)
            buildH = []
            while not clique_queue.empty():
                cur_stack = clique_queue.get()
                buildH.append([cur_stack.get()])
                if not cur_stack.empty():
                    clique_queue.put(cur_stack)
            self.sortedH = buildH
        elif sort_type == 'given':
            self.sortedH = [[h] for h in self.H]
        else:
            raise Exception('ERROR: unsupported sort_type: {}'.format(sort_type))
        # update the appropriate flags
        self.sort_type = sort_type
        self.H_withoutCo = self._getH_without_coef()


    def getExactUnitary(self, t):
        """
        Compute the exact unitary matrix for quantum evolution under the given
        Hamiltonian, self.H, for time t:

        U(t) = exp(-i * H * t)

        Using the scipy.linalg.expm function (https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html#scipy.linalg.expm)

        Parameters
        ----------
        t : float
            Evolution time

        Returns
        -------
        U(t) : numpy array
        """
        I = np.identity(2)
        X = np.array([[0,1],[1,0]], dtype='complex64')
        Y = np.array([[0,0-1j],[0+1j,0]], dtype='complex64')
        Z = np.array([[1,0],[0,-1]], dtype='complex64')
        Pdict = {'I':I, 'X':X, 'Y':Y, 'Z':Z}

        # H = ZI + ZZ
        # uexact = exp(-i * (ZI+ZZ) * t)

        Hmatrix = 0
        for term in self.H: # for each term in Hamiltonian
            # get the coefficient and Pauli string
            coef, pauli_str = term
            term_matrix = 1
            # compute the Kronecker product for the entire string
            for P in pauli_str[::-1]:
                # we begin at the end of the string and work towards the front
                # Example: XYZ = kron(X, kron(Y, kron(Z, 1)))
                term_matrix = np.kron(Pdict[P], term_matrix)
            # scale the result by coefficient and add to previous result
            Hmatrix = Hmatrix + coef * term_matrix
        # exponentiate the Hamiltonian
        U = linalg.expm((0-1j) * Hmatrix * t)
        return U.astype('complex64')


    def getAncillaUnitary(self, t):
        """
        Compute the exact unitary matrix for quantum evolution under the given
        Hamiltonian, self.H, for time t:

        U(t) = exp(-i * H * t)

        Using the scipy.linalg.expm function (https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html#scipy.linalg.expm)

        Parameters
        ----------
        t : float
            Evolution time

        Returns
        -------
        U(t) : numpy array
        """
        I = np.identity(2)
        X = np.array([[0,1],[1,0]], dtype='complex64')
        Y = np.array([[0,0-1j],[0+1j,0]], dtype='complex64')
        Z = np.array([[1,0],[0,-1]], dtype='complex64')
        Pdict = {'I':I, 'X':X, 'Y':Y, 'Z':Z}

        # H = ZI + ZZ
        # uexact = exp(-i * (ZI+ZZ) * t)

        Hmatrix = 0
        for term in self.H: # for each term in Hamiltonian
            # get the coefficient and Pauli string
            coef, pauli_str = term
            term_matrix = 1
            # compute the Kronecker product for the entire string
            for P in pauli_str[::-1]:
                # we begin at the end of the string and work towards the front
                # Example: XYZ = kron(X, kron(Y, kron(Z, 1)))
                term_matrix = np.kron(Pdict[P], term_matrix)
            term_matrix = np.kron(I, term_matrix)
            # scale the result by coefficient and add to previous result
            Hmatrix = Hmatrix + coef * term_matrix
        # exponentiate the Hamiltonian
        U = linalg.expm((0-1j) * Hmatrix * t)
        return U.astype('complex64')

    def getCircuitUnitary(self, iterateUnitary=False):
        """
        Compute the unitary implemented by the given circuit

        Parameters
        ----------
        iterateUnitary : bool
            If true, return the circuit unitary for a single iteration of the
            TS-decomposition

        Returns
        -------
        U : numpy array
            Matrix representation of the quantum circuit
        """
        unitarysimulator = Aer.get_backend('unitary_simulator')
        #return execute(circ, unitarysimulator).result().get_unitary(circ)

        # compute sum of the global phases appearing in a single TS iteration
        phase_sum = np.sum(self.deltaTs[1])

        # NOTE: the unitary_simulator backend has a zero_threshold parameter
        # which truncates small values to zero (default: 1e-10)
        if iterateUnitary:
            # Obtain the circuit unitary for a single TS iteration using the
            # Qiskit unitary_simulator backend
            print('Calculating iterateUnitary')
            #U = Operator(self.iterate_circ_copy).data
            result = execute(self.iterate_circ_copy, unitarysimulator).result()
            U = result.get_unitary(self.iterate_circ_copy)

            # Compute the necessary phase correction
            phase_correction = np.exp((0-1j) * phase_sum / 2)
        else:
            # Obtain the circuit unitary for the full TS decomposition using the
            # Qiskit unitary_simulator backend
            print('Calculating fullCircuitUnitary')
            #U = Operator(self.full_circ_copy).data
            result = execute(self.full_circ_copy, unitarysimulator).result()
            U = result.get_unitary(self.full_circ_copy)

            # Compute the necessary phase correction - the difference here is we
            # must multiply the phase_sum by the current value of r = deltaTs[0]
            # to account for the global phase which is accumlated for each
            # iteration of the TS decomposition
            phase_correction = np.exp((0-1j) * self.deltaTs[0] * phase_sum / 2)

        # Multiply U by the phase_correction
        #pc_U = phase_correction * U

        # After this correction, there may be real or imaginary values in pc_U
        # that are close to zero (O(1e-16)), we will manually set those values
        # to zero here, using the same threshold as Qiskit's unitary_simulator
        #threshold = 1e-10
        #pc_U.real[abs(pc_U.real) < threshold] = 0.0
        #pc_U.imag[abs(pc_U.imag) < threshold] = 0.0
        #return pc_U.astype('complex64')
        return U.astype('complex64')


    def compute2Norm(self, U1, U2):
        """
        Compute the 2-norm distance between matrices U1 and U2

        Parameters
        ----------
        U1 : numpy array
        U2 : numpy array

        Returns
        -------
        The 2-norm distance between U1 and U2:

        || U1 - U2 ||
        """
        return np.linalg.norm((U1-U2), ord=2)


    def mathematicaFidelity(self, Uexact, Uapprox):
        """
        Compute the fidelity between Uexact and Uapprox.

        Fidelity = |Tr(Uexact . Uapprox^dag)| / numrows(Uexact)
        """
        return abs(np.trace(np.matmul(Uexact, Uapprox.conj().T))) / Uexact.shape[0]


    def mathematicaInfidelity(self, Uexact, Uapprox):
        """
        Compute the infidelity between Uexact and Uapprox.

        Infidelity = 1 - mathematicaFidelity(Uexact, Uapprox)
        """
        return 1 - self.mathematicaFidelity(Uexact, Uapprox)


    def processFidelity(self, Uexact, Uapprox):
        """
        Compute the process fidelity between the circuit and exact unitaries.

            F(e,u) = Tr[S_u^dag.S_e] / d^2

        Where S_e and S_u are SuperOperators for Uexact and Uapprox, and d is
        the dimension of the channel
        """
        # Generate target Unitary
        target = Operator(Uexact)

        # Generate the input quantum channel
        # If the circuit includes an ancilla, we must select the 0-subspace of
        # the ancilla
        exactRows, exactCols = Uexact.shape
        circRows, circCols = Uapprox.shape
        if exactRows == circRows and exactCols == circCols:
            # the two unitaries have the same dimension -> no ancilla was added
            anc0subspace = Uapprox
        elif exactRows * 2 == circRows and exactCols * 2 == circCols:
            # Circuit dimension is twice the size of Uexact dimension -> ancilla
            # was added
            anc0subspace = Uapprox[:(circRows // 2),:(circCols // 2)]
        else:
            # Something is wrong
            raise Exception('Dimension error: dim(Uexact) = {}, dim(Uapprox) = {}'.format(Uexact.shape, Uapprox.shape))

        channel = SuperOp(Operator(anc0subspace))
        return round(process_fidelity(channel, target=target), 4)


if __name__ == '__main__':
    hfile = 'hamiltonians/H2_6-31g_BK_0.7_AS2.txt'
    #hfile = 'EXACT: (1)IZ'
    dynamics = Dynamics(hfile)
    #print('dynamics.H =',dynamics.H)
    t = 1
    Uexact = dynamics.getExactUnitary(t)
    #print(type(Uexact), Uexact.dtype)
    #print('Uexact:\n', np.matrix(Uexact))
    #print('H:\n',dynamics.H_withoutCo)
    dynamics.sort_hamiltonian(sort_type='all_tsp', print_info=True)
    print('sortedH:\n',dynamics.sortedH)
    #print('HwithoutCo = ',dynamics.H_withoutCo)
    #circ = dynamics.gen_circuit(t, r=1, transpile=None, func='MikeAndIke', mode='star', barriers=True)
    #print(circ)
    #circ = dynamics.gen_circuit(t, r=1, transpile=None, func='MikeAndIke', mode='ladder', barriers=True)
    #print(circ)
    #Uapprox = dynamics.getCircuitUnitary()
    #print(type(Uapprox), Uapprox.dtype)
    #print('Uapprox:\n', np.matrix(Uapprox))
    #print('2norm distance = {}'.format(dynamics.compute2Norm(Uexact,Uapprox)))
    #print('fidelity = {}'.format(dynamics.mathematicaFidelity(Uexact,Uapprox)))

    #circ.draw(filename='figures/testing_H2_dynamics_circuit.txt', output='text',
    #          line_length=200)
