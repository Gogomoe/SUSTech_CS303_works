import argparse
from typing import Optional, List, Tuple, Set
from random import randint, random, choice
from math import log, log2, sqrt, e
from heapdict import heapdict
from diffusion import Node, Edge, DiffusionModel, ICModel, LTModel
from time import time


class CELFNode:
    def __init__(self, node: int):
        self.node: int = node
        self.mg1: float = 0
        self.mg2: float = 0
        self.prev_best: Optional[CELFNode] = None
        self.flag: Optional[int] = None


model: str

diffusion_model: DiffusionModel

diffuse_times: int = 10

start_time: float
time_limit: int


def CELF(network: List[Node], k: int) -> List[int]:
    global diffusion_model

    S = set()
    Q = heapdict()

    last_seed: Optional[CELFNode] = None
    cur_best: Optional[CELFNode] = None
    node_data_list: List[CELFNode] = []

    nodes = len(network) - 1

    node_data_list.append(None)

    for node in range(1, nodes + 1):
        node_data: CELFNode = CELFNode(node)
        node_data.mg1 = diffusion_model.diffuse_times([node], diffuse_times) / float(diffuse_times)
        if cur_best:
            node_data.prev_best = cur_best
            node_data.mg2 = diffusion_model.diffuse_times([node, cur_best.node], diffuse_times) / float(diffuse_times)
        else:
            node_data.prev_best = None
            node_data.mg2 = node_data.mg1
        node_data.flag = 0
        node_data_list.append(node_data)
        Q[node] = - node_data.mg1
        cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
    while len(S) < k:
        node, _ = Q.peekitem()
        node_data = node_data_list[node]
        if node_data.flag == len(S):
            S.add(node)
            del Q[node]
            last_seed = node_data
            continue
        elif node_data.prev_best == last_seed:
            node_data.mg1 = node_data.mg2
        else:
            before: float = diffusion_model.diffuse_times(list(S), diffuse_times) / float(diffuse_times)
            S.add(node)
            after: float = diffusion_model.diffuse_times(list(S), diffuse_times) / float(diffuse_times)
            S.remove(node)

            node_data.mg1 = after - before
            node_data.prev_best = cur_best

            S.add(cur_best.node)
            before: float = diffusion_model.diffuse_times(list(S), diffuse_times) / float(diffuse_times)
            S.add(node)
            after: float = diffusion_model.diffuse_times(list(S), diffuse_times) / float(diffuse_times)
            S.remove(cur_best.node)
            if node != cur_best.node:
                S.remove(node)
            node_data.mg2 = after - before
        if cur_best and cur_best.mg1 < node_data.mg1:
            cur_best = node_data

        node_data.flag = len(S)
        Q[node] = -node_data.mg1
    return list(S)


def logcnk(n, k):
    res = 0
    for i in range(n - k + 1, n + 1):
        res += log(i)
    for i in range(1, k + 1):
        res -= log(i)
    return res


def generate_rr(network: List[Node], v: int) -> List[int]:
    global model
    if model == 'IC':
        return generate_rr_ic(network, v)
    elif model == 'LT':
        return generate_rr_lt(network, v)


def generate_rr_ic(network: List[Node], v: int) -> List[int]:
    activity_set: List[int] = [v]
    activity_nodes: List[int] = [v]
    while activity_set:
        new_activity_set: List[int] = []
        for seed in activity_set:
            for edge in network[seed].ins:
                if edge.fo not in activity_nodes:
                    if random() < edge.weight:
                        activity_nodes.append(edge.fo)
                        new_activity_set.append(edge.fo)
        activity_set = new_activity_set
    return activity_nodes


def generate_rr_lt(network: List[Node], v: int) -> List[int]:
    activity_nodes: List[int] = [v]
    activity_set: int = v

    while activity_set != -1:
        new_activity_set = -1

        neighbors = network[activity_set].ins
        if len(neighbors) == 0:
            break
        candidate = choice(neighbors).fo
        if candidate not in activity_nodes:
            activity_nodes.append(candidate)
            new_activity_set = candidate
        activity_set = new_activity_set
    return activity_nodes


def node_selection(network: List[Node], R: List[List[int]], k: int) -> Tuple[Set[int], float]:
    S_k: Set[int] = set()
    rr_degree = [0] * len(network)
    node_rr_set = dict()
    matched_count = 0
    for i in range(len(R)):
        rr: List[int] = R[i]
        for rr_node in rr:
            rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
            node_rr_set[rr_node].append(i)

    for i in range(k):
        max_point = rr_degree.index(max(rr_degree))
        S_k.add(max_point)
        matched_count += len(node_rr_set[max_point])
        index_set = []
        for node_rr in node_rr_set[max_point]:
            index_set.append(node_rr)
        for j in index_set:
            rr = R[j]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(j)
    return S_k, matched_count / len(R)


def sampling(network: List[Node], k: int, epsilon: float, l: float) -> List[List[int]]:
    n: int = len(network) - 1

    R: List[List[int]] = []
    LB: float = 1.0
    epsilon_p = sqrt(2) * epsilon

    for i in range(1, int(log2(n - 1)) + 1):
        x = n / 2 ** i
        lambda_p = ((2 + 2 / 3 * epsilon_p) * (logcnk(n, k) + l * log(n) + log(log2(n))) * n) / (epsilon_p ** 2)
        theta_i = lambda_p / x

        while len(R) <= theta_i:
            v: int = randint(1, n)
            rr: List[int] = generate_rr(network, v)
            R.append(rr)
        S_i: Set[int]
        F_R: float
        S_i, F_R = node_selection(network, R, k)
        if n * F_R >= (1 + epsilon_p) * x:
            LB = n * F_R / (1 + epsilon_p)
            break
    alpha = sqrt(l * log(n) + log(2))
    beta = sqrt((1 - 1 / e) * (logcnk(n, k) + l * log(n) + log(2)))
    lambda_star = 2 * n * (((1 - 1 / e) * alpha + beta) ** 2) * (epsilon ** -2)
    theta = lambda_star / LB
    while (len(R)) <= theta:
        v: int = randint(1, n)
        rr: List[int] = generate_rr(network, v)
        R.append(rr)
    return R


def IMM(network: List[Node], k: int, epsilon: float, l: float) -> List[int]:
    n: int = len(network) - 1
    l = l * (1 + log(2) / log(n))
    # R: List[List[int]] = sampling(network, k, epsilon, l)
    R: List[List[int]] = []

    while time() - start_time < time_limit * 0.2 and len(R) < 5_0000:
        v: int = randint(1, n)
        R.append(generate_rr(network, v))

    S_k: Set[int]
    S_k, _ = node_selection(network, R, k)

    return list(S_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-k', '--size', type=int, default='0')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name: str = args.file_name
    size: int = args.size
    global model, time_limit
    model = args.model
    time_limit = args.time_limit

    nodes: int
    edges: int

    network: List[Node] = [Node(0)]
    seeds: List[int] = []

    with open(file_name, 'r') as reader:
        line: str = reader.readline()
        split = line.split()

        nodes = int(split[0])
        edges = int(split[1])

        for i in range(1, nodes + 1):
            network.append(Node(i))

        for i in range(edges):
            line = reader.readline()
            split = line.split()
            edge = Edge(int(split[0]), int(split[1]), float(split[2]))
            network[edge.fo].out.append(edge)
            network[edge.to].ins.append(edge)

    global diffusion_model
    if model == "IC":
        diffusion_model = ICModel(network)
    elif model == "LT":
        diffusion_model = LTModel(network)

    # result: List[int] = CELF(network, size)
    result: List[int] = IMM(network, size, 0.5, 1.0)

    for i in result:
        print(i)


if __name__ == '__main__':
    start_time: float = time()
    main()
