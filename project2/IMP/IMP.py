import argparse
from typing import Optional, List

from heapdict import heapdict
from diffusion import Node, Edge, DiffusionModel, ICModel, LTModel


class CELFNode:
    def __init__(self, node: int):
        self.node: int = node
        self.mg1: float = 0
        self.mg2: float = 0
        self.prev_best: Optional[CELFNode] = None
        self.flag: Optional[int] = None


diffuse_times: int = 20


def CELF(network: List[Node], diffusion_model: DiffusionModel, k: int) -> List[int]:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-k', '--size', type=int, default='0')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name: str = args.file_name
    size: int = args.size
    model: str = args.model
    time_limit: int = args.time_limit

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

    diffusion_model: DiffusionModel = None
    if model == "IC":
        diffusion_model = ICModel(network)
    elif model == "LT":
        diffusion_model = LTModel(network)

    result: List[int] = CELF(network, diffusion_model, size)
    for i in result:
        print(i)


if __name__ == '__main__':
    main()
