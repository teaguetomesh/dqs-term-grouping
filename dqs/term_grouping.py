# -*- coding: utf-8 -*-
"""
Given a particular qubit Hamiltonian, measuring the expected energy of any
given quantum state will depend only on the individual terms of that
Hamiltonian.

measureCircuit.py generates a circuit which will measure a quantum state in the
correct bases to allow the energy to be calculated. This may require generating
multiple circuits if the same qubit needs to be measured in two perpendicular
bases (i.e. Z and X).

To find the minimum number of circuits needed to measure an entire Hamiltonian,
we treat the terms of H as nodes in a graph, G, where there are edges between
nodes indicate those two terms commute with one another. Finding the circuits
now becomes a clique finding problem which can be solved by the
BronKerbosch algorithm.
"""


import time
import sys
import numpy as np
import pprint
import networkx as nx
from networkx.algorithms import approximation
import mlrose


class CommutativityType(object):
    def gen_comm_graph(term_array):
        raise NotImplementedError


class QWCCommutativity(CommutativityType):
    def gen_comm_graph(term_array):
        g = {}
        for i, term1 in enumerate(term_array):
            comm_array = []
            for j, term2 in enumerate(term_array):

                if i == j: continue
                commute = True
                for c1, c2 in zip(term1, term2):
                    if c1 == '*': continue
                    if (c1 != c2) and (c2 != '*'):
                        commute = False
                        break
                if commute:
                    comm_array += [''.join(term2)]
            g[''.join(term1)] = comm_array

        #print('TERMGROUPING: Generated graph for the Hamiltonian with {} nodes.'.format(len(g)))

        return g


class GeneralCommutativity(CommutativityType):
    def gen_comm_graph(term_array):
        g = {}

        for i, term1 in enumerate(term_array):
            comm_array = []
            for j, term2 in enumerate(term_array):

                if i == j: continue
                non_comm_indices = 0
                for c1, c2 in zip(term1, term2):
                    if c1 == '*': continue
                    if (c1 != c2) and (c2 != '*'):
                        non_comm_indices += 1
                if (non_comm_indices % 2) == 0:
                    comm_array += [''.join(term2)]
            g[''.join(term1)] = comm_array

        #print('TERMGROUPING: Generated graph for the Hamiltonian with {} nodes.'.format(len(g)))

        return g


def getCommClass(commtype):
    if commtype.lower() == 'gc':
        commClass = GeneralCommutativity
    elif commtype.lower() == 'qwc':
        commClass = QWCCommutativity
    else:
        raise Exception('Unknown Commutativity Class:', commtype)
    return commClass


def prune_graph(G,nodes):
    for n in nodes:
        neighbors = G.pop(n)
        for nn in neighbors:
            G[nn].remove(n)

def degeneracy_ordering(graph):
    """
    Produce a degeneracy ordering of the vertices in graph, as outlined in,
    Eppstein et. al. (arXiv:1006.5440)
    """

    # degen_order, will hold the vertex ordering
    degen_order = []

    while len(graph) > 0:
        # Populate D, an array containing a list of vertices of degree i at D[i]
        D = []
        for node in graph.keys():
            Dindex = len(graph[node])
            cur_len = len(D)
            if cur_len <= Dindex:
                while cur_len <= Dindex:
                    D.append([])
                    cur_len += 1
            D[Dindex].append(node)

        # Add the vertex with lowest degeneracy to degen_order
        for i in range(len(D)):
            if len(D[i]) != 0:
                v = D[i].pop(0)
                degen_order += [v]
                prune_graph(graph,[v])

    return degen_order


def degree_ordering(G):
    nodes = list(G.keys())
    return sorted(nodes, reverse=True, key=lambda n: len(G[n]))


def BronKerbosch_pivot(G,R,P,X,cliques):
    """
    For a given graph, G, find a maximal clique containing all of the vertices
    in R, some of the vertices in P, and none of the vertices in X.
    """
    if len(P) == 0 and len(X) == 0:
        # Termination case. If P and X are empty, R is a maximal clique
        cliques.append(R)
    else:
        # choose a pivot vertex
        pivot = next(iter(P.union(X)))
        # Recurse
        for v in P.difference(G[pivot]):
            # Recursion case. 
            BronKerbosch_pivot(G,R.union({v}),P.intersection(G[v]),
                               X.intersection(G[v]),cliques)
            P.remove(v)
            X.add(v)


def BronKerbosch(G):
    """
    Implementation of Bron-Kerbosch algorithm (Bron, Coen; Kerbosch, Joep (1973),
    "Algorithm 457: finding all cliques of an undirected graph", Commun. ACM,
    ACM, 16 (9): 575–577, doi:10.1145/362342.362367.) using a degree ordering
    of the vertices in G instead of a degeneracy ordering.
    See: https://en.wikipedia.org/wiki/Bron-Kerbosch_algorithm
    """

    max_cliques = []

    while len(G) > 0:
        P = set(G.keys())
        R = set()
        X = set()
        v = degree_ordering(G)[0]
        cliques = []
        BronKerbosch_pivot(G,R.union({v}),P.intersection(G[v]),
                           X.intersection(G[v]),cliques)

        #print('i = {}, current v = {}'.format(i,v))
        #print('# cliques: ',len(cliques))

        sorted_cliques = sorted(cliques, key=len, reverse=True)
        max_cliques += [sorted_cliques[0]]
        #print(sorted_cliques[0])

        prune_graph(G,sorted_cliques[0])

    return max_cliques


def Boppana(graph_dict):
    """
    NetworkX poly-time heuristic is based on
    Boppana, R., & Halldórsson, M. M. (1992).
    Approximating maximum independent sets by excluding subgraphs.
    BIT Numerical Mathematics, 32(2), 180–196. Springer.
    """
    G = nx.Graph()
    for src in graph_dict:
        G.add_node(src)
        for dst in graph_dict[src]:
            G.add_edge(src, dst)
    return approximation.clique_removal(G)[1]


def _initializeA_U(G, U):
    """Initialize A_U
    A_U(x) = # of neighbors in U for node x in U
    """
    A_U = {}
    for x in U:
        all_neighbors = list(G[x])
        count = 0
        for node in all_neighbors:
            if node in U:
                count += 1
        A_U[x] = count
    return A_U


def _maxAUNode(U, A_U):
    """Find the node in U with the largest A_U value"""
    if len(U) == 0:
        raise Exception('All nodes are colored! This function should not be called')

    highest_val = -1
    for x in U:
        if A_U[x] > highest_val:
            highest_val = A_U[x]
            ret_node = x
    return ret_node


def _maxAWNode(U, A_W, A_U):
    """Find the node in U with the largest A_W value"""
    highest_val = -1
    for u in U:
        if A_W[u] > highest_val:
            highest_val = A_W[u]
            ret_node = u
        elif A_W[u] == highest_val:
            if A_U[u] < A_U[ret_node]:
                highest_val = A_W[u]
                ret_node = u
    return ret_node


def _addToColorClass(G, C_v, U, W, A_U, A_W, u):
    """Add u to the color class and move its neighbors to W"""
    # move u from U to C_v
    C_v.add(u)
    U.remove(u)
    # get the neighbors of u which are in U
    u_neighbors_in_U = U.intersection(G[u])
    for w in u_neighbors_in_U:
        # decrement A_U[w] for each of those neighbors
        A_U[w] -= 1
        # move those neighbors from U to W
        W.add(w)
        U.remove(w)
        # get the neighbors of w which are in U
        w_neighbors_in_U = U.intersection(G[w])
        for x in w_neighbors_in_U:
            # decrement A_U[x] for each of those neighbors
            A_U[x] -= 1
            # increment A_W[x] for each of those neighbors
            A_W[x] += 1


def _constructColorClass(G, U, v, A_U):
    """Construct the color class C_v"""
    # Initialize W and C_v
    # W = set of uncolored vertices with at least one neighbor in C_v
    # C_v = the color class containing v
    W, C_v = set(), set()
    # Initialize A_W, A_W[x] = # of neighbors that x has in W
    A_W = {}
    for x in U:
        A_W[x] = 0

    # start by adding v to C_v
    _addToColorClass(G, C_v, U, W, A_U, A_W, v)

    # while U is not empty
    while len(U) > 0:
        u = _maxAWNode(U, A_W, A_U)
        # Move u to C_v and all neighbors to W
        _addToColorClass(G, C_v, U, W, A_U, A_W, u)

    # Once U is empty, the number of colorable nodes has been exhausted
    return C_v


def RLF(graph_dict):
    """
    Implementation of Recursive Largest First (RLF) algorithm from
    Leighton, F.T. 1979 for graph coloring. Also mentioned in term-grouping
    work by Verteletskyi, Yen, Izmaylov (https://arxiv.org/abs/1907.03358)

    The min-clique-cover problem for a graph G can be mapped to a coloring
    problem on the complement of G (Gbar). The chromatic number of Gbar is equal
    to the min-clique-cover of G. Each clique corresponds to nodes of Gbar which
    have the same color.

    This implementation is based on the description in Adegbindin, M., Hertz,
    A. and BELLA, M., 2016. A new efficient RLF-like Algorithm for the Vertex
    Coloring Problem. Yugoslav Journal of Operations Research, 26(4).

    Parameters
    ----------
    graph_dict : dict
        Dictionary where the keys are nodes in the graph and the values are
        lists of the key's neighbors

    Returns
    -------
    A minimum-clique-cover of the graph G
    """
    # generate a networkx graph for the given Hamiltonian
    G = nx.Graph()
    for src in graph_dict:
        G.add_node(src)
        for dst in graph_dict[src]:
            G.add_edge(src, dst)

    # Coloring the complement of G will provide a min-clique-cover
    Gbar = nx.complement(G)

    # store the coloring in a list of lists where the list at index i contains
    # all nodes in color class i, where i=0 is the uncolored class
    color = [list(Gbar.nodes)]
    while len(color[0]) > 0:
        # Construct U = the set of uncolored vertices
        U = set(color[0])

        # Initialize A_U
        A_U = _initializeA_U(Gbar, U)

        # Choose vertex in U with largest value of A_U(v)
        v = _maxAUNode(U, A_U)

        # Construct C_v = the color class containing v
        C_v = _constructColorClass(Gbar, U, v, A_U)

        # Color all nodes in C_v (i.e. add the color class C_v to the color list
        # at index i
        color[0] = [node for node in color[0] if node not in C_v]
        color.append(C_v)

    # take the generated coloring and return a min-clique-cover
    return color[1:]


def getCoverFunc(clique_cover_method):
    if clique_cover_method.lower() in ['rlf']:
        coverFunc = RLF
    elif clique_cover_method.lower() in ['boppana', 'bh']:
        coverFunc = Boppana
    elif clique_cover_method.lower() in ['bronk', 'bronkerbosch', 'bk']:
        coverFunc = BronKerbosch
    else:
        raise Exception('Unknown cover function:', clique_cover_method)
    return coverFunc


def sortH(H, grouped_terms, mode, gate_cancellation=False, print_info=False):
    """
    Resort the given H according to the order given in grouped_terms.
    If gate_cancellation, attempt to maximize amount of gate cancellation.
    """
    sortedH = []
    for n, group in enumerate(grouped_terms):
        if gate_cancellation:
            if print_info:
                print('Clique', n)
            group = orderGroupForGateCancellation(group, mode, print_info=print_info)
        new_group = []
        for term in group:
            for i, hterm in enumerate(H):
                if term == hterm[1]:
                    new_group.append(hterm)
                    break
        sortedH.append(new_group)
    #print('sortedH:\n',sortedH)
    clean_sortedH = []
    for cliq in sortedH:
        clean_cliq = [(float(term[0]), term[1]) for term in cliq]
        clean_sortedH.append(clean_cliq)
    return clean_sortedH


def sort_all_tsp(H, mode, print_info=True):
    """
    Order the entire Hamiltonian according to the TSP heuristic
    """
    sortedH = []
    tspOrder = orderGroupForGateCancellation([term[1] for term in H], mode,
                                              print_info=print_info)
    for term in tspOrder:
        for hterm in H:
            if term == hterm[1]:
                sortedH.append((float(hterm[0]), hterm[1]))
                break
    return sortedH


def orderGroupForGateCancellation(group, mode, print_info=True):
    """
    Orders terms within a group (clique) such that the number of possible gate
    cancellations is maximized.

    Uses the genetic_algorithm supplied by mlrose to produce a Travelling
    Salesperson (TSP) path through the graph.

    Produce a Hamiltonian Cycle through the graph by deleting the most expensive
    edge in the TSP path.

    Parameters
    ----------
    group : List(str)
        a clique of Pauli strings
    print_info : bool
        print extra info on Hamiltonian Cycles found

    Returns
    -------
    List(str)
        a clique of Pauli strings where the ordering of the strings maximizes
        gate cancellations
    """
    if len(group) == 1:
        if print_info:
            print('Single term in group, returning: ', list(group))
        return list(group)

    # Find a TSP path using the mlrose module
    group = list(group)
    mlrose_distances, distance_matrix = _pairwise_distances(group, mode=mode)
    problem = mlrose.TSPOpt(length=len(group), distances=mlrose_distances)
    mlrose_best_state, mlrose_best_fitness  = mlrose.genetic_alg(problem)

    # Use the distance_matrix to break the most expensive edge in the TSP path,
    # giving a Hamiltonian Cycle with lower overall distance
    path_distances = []
    for i in range(len(mlrose_best_state)):
        # select a node in the path
        this_node = mlrose_best_state[i]
        # select the next node in the path
        if i == len(mlrose_best_state)-1:
            # if this_node is the last node, loop back to the beginning
            next_node = mlrose_best_state[0]
        else:
            next_node = mlrose_best_state[i+1]
        # add the distance from this_node to next_node to path_distances
        path_distances.append(distance_matrix[max(this_node,next_node),
                                              min(this_node,next_node)])

    # find the most expensive edge
    largest_dist_seen = -1
    for i, dist in enumerate(path_distances):
        if dist >= largest_dist_seen:
            largest_dist_seen = dist
            expensive_edge = i

    # given the most expensive edge, slice the mlrose_best_state
    mlrose_best_state = list(mlrose_best_state)
    best_state = mlrose_best_state[expensive_edge+1:] + mlrose_best_state[:expensive_edge+1]
    best_fitness = np.sum(path_distances) - path_distances[expensive_edge]

    # Use best_state to order the terms within the group
    ordered_group = [group[i] for i in best_state]

    if print_info:
        print(' '*3,'original order was %s' % group)
        print(' '*3,'new order is %s' % ordered_group)
        print(' '*3,'total Hamiltonian Cycle distance is %s' % best_fitness)

    return ordered_group


def _pairwise_distances(group, mode='star'):
    """Pairwise CNOT distances between Pauli strings in group, used for TSP heuristic."""
    mlrose_distances = []
    distance_matrix = np.zeros((len(group),len(group)), dtype=int)
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            term1, term2 = group[i], group[j]
            if mode == 'star':
                cur_distance = _star_distance(term1, term2)
            elif mode == 'ladder':
                cur_distance = _ladder_distance(term1, term2)
            # mlrose distances are given as a list of triples [(u,v,d),...]
            mlrose_distances.append((i, j, cur_distance))
            # We also store the distances in a lower-triangular matrix
            distance_matrix[j,i] = cur_distance
    return mlrose_distances, distance_matrix


def _ladder_distance(term1, term2):
    """naive ladder CNOT distance between two Pauli strings, used for TSP Heuristic."""
    assert len(term1) == len(term2), '%s and %s have different lengths' % (term1, term2)
    assert all([char in ['I', 'X', 'Y', 'Z'] for char in term1]), '%s has non IXYZ chars' % term1
    assert all([char in ['I', 'X', 'Y', 'Z'] for char in term2]), '%s has non IXYZ chars' % term2

    # First calculate the total required CNOTs, then deduct the cancellation.
    # For each Pauli string, total required CNOTs equals the number of non-I terms minus 1.
    # Edge case happens when there is zero or one non-I character, already handled
    total_non_I_term1 = 0
    total_non_I_term2 = 0
    term1_CNOT = 0
    term2_CNOT = 0
    for i in range(len(term1)):
        if (term1[i] != 'I'):
            total_non_I_term1 += 1
        if (term2[i] != 'I'):
            total_non_I_term2 += 1
    if total_non_I_term1 > 1:
        term1_CNOT = total_non_I_term1 - 1
    if total_non_I_term2 > 1:
        term2_CNOT = total_non_I_term2 - 1

    # Next do deduction
    # If the right most and 2nd right most characters are the same, then we can cancel the first outer layer CNOT pair.
    # Cancel layer by layer after, stop when characters start to differ.
    # Start iterating from the right most Pauli character.
    # Edge case happens when there is zero or one same character, already handled
    same_count = 0
    CNOT_reduction = 0
    for i in range(len(term1)):
        reverse_iterator = len(term1) - 1 - i
        # print(reverse_iterator)
        if term1[reverse_iterator] != term2[reverse_iterator]:
            # if there is any different Pauli characters, break the loop
            break
        elif (term1[reverse_iterator] != 'I' and term2[reverse_iterator] != 'I'):
            # for other non-I same character pairs, increment the same_count by 1
            same_count += 1
    if same_count > 1:
        CNOT_reduction = (same_count - 1) * 2
    # elif same_count == 1 or same_count == 0, CNOT_reduction stays 0

    # print('term1 has CNOT # = ', term1_CNOT)
    # print('term2 has CNOT # = ', term2_CNOT)
    # Increase all ladder distances by 1, because some terms will have
    # a distance = 0 which is really good, but mlrose requires that
    # all distances are > 0
    return term1_CNOT + term2_CNOT - CNOT_reduction + 1


def _star_distance(term1, term2):
    """star + ancilla CNOT distance between two Pauli strings, used for TSP heuristic."""
    assert len(term1) == len(term2), '%s and %s have different lengths' % (term1, term2)
    assert all([char in ['I', 'X', 'Y', 'Z'] for char in term1]), '%s has non IXYZ chars' % term1
    assert all([char in ['I', 'X', 'Y', 'Z'] for char in term2]), '%s has non IXYZ chars' % term2

    # edge case: if there is only zero or one none-I term (e.g. IIIIXIII) in either of the two terms
    # then at least one term have no CNOT gate, and therefore nothing to cancel
    # (this is very unlikely, but under the star + ancilla implementation this could create a little unnecessary CNOT overhead)

    # general case
    # According to Eq.14 in https://arxiv.org/pdf/2001.05983.pdf
    # total of 4 x 4 = 16 XYZI letter combinations
    # for XY, XZ, YX, YZ, ZX, ZY, 2 CNOTs will incur (distance + 2)
    # for IX, IY, IZ, XI, YI, ZI, 1 CNOT will incur (distance + 1)
    # for XX, YY, ZZ everything cancel perfectly (distance + 0)
    # for II, nothing needed (distance + 0)
    distance = 0
    for i in range(len(term1)):
        if term1[i] != term2[i]:
            distance += 2
        if (term1[i] == 'I' and term2[i] == 'X') or (term1[i] == 'I' and term2[i] == 'Y') or (term1[i] == 'I' and term2[i] == 'Z') or (term1[i] == 'X' and term2[i] == 'I') or (term1[i] == 'Y' and term2[i] == 'I') or (term1[i] == 'Z' and term2[i] == 'I'):
            distance -= 1
    return distance


def findMinCliqueCover(H, Nq, commutativity_type='gc', clique_cover_method='bk',
                       gate_cancellation=False, print_info=True, mode='star'):
    """
    For a given Hamiltonian find the minimum number of maximal cliques to cover
    the associated graph.

    Parameters
    ----------
    H : List[(coef,pauli_ops)]
        Representation of the Hamiltonian where each list entry stores the
        coefficient and pauli operators for a single term
    Nq : int
        Number of qubits
    commutativity_type : str
        Select either the QWCCommutativity or the GeneralCommutativity class
    clique_cover_method : str
        Select the clique finding algorithm
    gate_cancellation : bool
        Should terms within cliques be ordered according to a TSP optimization?
    print_info : bool
        Print info during execution

    Returns
    -------
    max_cliques : List[Hamiltonian terms]
        A list where each entry contains a set of mutually commuting terms in
        Hamiltonian. The size of this list is the MinCliqueCover for this graph.
    """
    commClass = getCommClass(commutativity_type)
    coverFunc = getCoverFunc(clique_cover_method)

    start_time = time.time()

    term_reqs = np.full((len(H), Nq),'*',dtype=str)
    for i, term in enumerate(H):
        for j, op in enumerate(term[1]):
            if op == 'I': op = '*'
            term_reqs[i][j] = op
    #print('term_reqs:\n',term_reqs)

    # Generate a graph representing the commutativity of the Hamiltonian terms
    comm_graph = commClass.gen_comm_graph(term_reqs)

    # Find a set of cliques within the graph where the nodes in each clique
    # are disjoint from one another.
    max_cliques = coverFunc(comm_graph)

    end_time = time.time()

    #print('TERMGROUPING: {} found MIN_CLIQUE_COVER = {}'.format(
    #    coverFunc.__name__, len(max_cliques)))
    et = end_time - start_time
    #print('TERMGROUPING: Elapsed time: {:.6f}s'.format(et))

    # resort the Hamiltonian according to the cliques in max_cliques
    #print('max_cliques:\n',max_cliques)
    max_cliques_clean = []
    for group in max_cliques:
        new_group = set()
        for term in group:
            new_group.add(term.replace('*','I'))
        max_cliques_clean.append(new_group)
    #print('max_cliques_clean:\n', max_cliques_clean)

    return sortH(H, max_cliques_clean, mode, gate_cancellation=gate_cancellation, print_info=print_info)


def benchmarkMinCliqueCover(H, Nq, commutativity_type='gc',
                            clique_cover_method='rlf'):
    """
    Same function as findMinCliqueCover, but here we return the number of nodes
    in the generated graph, the min-clique-cover, and the time elapsed
    """
    commClass = getCommClass(commutativity_type)
    coverFunc = getCoverFunc(clique_cover_method)

    start_time = time.time()

    term_reqs = np.full((len(H), Nq),'*',dtype=str)
    for i, term in enumerate(H):
        for j, op in enumerate(term[1]):
            if op == 'I': op = '*'
            term_reqs[i][j] = op

    # Generate a graph representing the commutativity of the Hamiltonian terms
    comm_graph = commClass.gen_comm_graph(term_reqs)
    num_nodes = len(comm_graph)

    # Find a set of cliques within the graph where the nodes in each clique
    # are disjoint from one another.
    max_cliques = coverFunc(comm_graph)

    end_time = time.time()

    print('TERMGROUPING: {} found MIN_CLIQUE_COVER = {}'.format(
        coverFunc.__name__, len(max_cliques)))
    et = end_time - start_time
    print('TERMGROUPING: Elapsed time: {:.6f}s'.format(et))

    return num_nodes, len(max_cliques), et


if __name__ == "__main__":
    from utils import parse_hamiltonian_file as phf
    # change the number of qubits based on which hamiltonian is selected
    hfile = 'hamiltonians/H2_w_coef_bad_order.txt'
    Nq, H = phf.parseHfile(hfile)

    cliques = findMinCliqueCover(H, Nq, 'gc', 'rlf', gate_cancellation=True, print_info=True)
    for cliq in cliques:
        print(cliq)
