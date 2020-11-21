import argparse
from multiprocessing.pool import ApplyResult
from typing import List
import sys
import multiprocessing
import random

core = 8
N = 10000


class Edge:
    def __init__(self, fo: int, to: int, weight: float) -> None:
        self.fo = fo
        self.to = to
        self.weight = weight


class Node:
    def __init__(self, number: int):
        self.number = number
        self.out: List[Edge] = []
        self.ins: List[Edge] = []


class DiffusionModel:
    def __init__(self, network: List[Node]):
        self.network: List[Node] = network

    def diffuse(self, seeds: List[int]) -> int:
        pass

    def diffuse_times(self, seeds: List[int], times: int) -> int:
        result = 0
        for _ in range(times):
            result += self.diffuse(seeds)
        return result


class ICModel(DiffusionModel):
    def __init__(self, network: List[Node]):
        super().__init__(network)

    def diffuse(self, seeds: List[int]) -> int:
        network = self.network
        active: List[int] = seeds
        active_table: List[bool] = [False] * (len(network))

        result: int = len(seeds)
        for i in seeds:
            active_table[i] = True

        while len(active) != 0:
            next_active: List[int] = []
            for node in active:
                for edge in network[node].out:
                    if active_table[edge.to]:
                        continue
                    p: float = random.random()
                    if p <= edge.weight:
                        next_active.append(edge.to)
                        active_table[edge.to] = True
                        result += 1
            active = next_active

        return result


class LTModel(DiffusionModel):
    def __init__(self, network: List[Node]):
        super().__init__(network)

    def diffuse(self, seeds: List[int]) -> int:
        network = self.network
        active: List[int] = seeds
        active_table: List[bool] = [False] * (len(network))
        threshold: List[float] = []

        result: int = len(seeds)
        for i in seeds:
            active_table[i] = True

        for i in range(len(network)):
            threshold.append(random.random())

        while len(active) != 0:
            next_active: List[int] = []
            for node in active:
                for edge in network[node].out:
                    to_node = edge.to
                    if active_table[to_node]:
                        continue
                    w: float = 0.0
                    for x in network[to_node].ins:
                        if active_table[x.fo]:
                            w += x.weight

                    if w >= threshold[to_node]:
                        next_active.append(to_node)
                        active_table[to_node] = True
                        result += 1
            active = next_active

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-s', '--seed', type=str, default='seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name: str = args.file_name
    seed_file: str = args.seed
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

    with open(seed_file, 'r') as reader:
        for line in reader:
            seeds.append(int(line))

    pool = multiprocessing.Pool(core)
    result: List[ApplyResult] = []

    sum_result = 0

    diffusion_model: DiffusionModel = None
    if model == "IC":
        diffusion_model = ICModel(network)
    elif model == "LT":
        diffusion_model = LTModel(network)

    for i in range(core):
        result.append(pool.apply_async(diffusion_model.diffuse_times, args=(seeds, N // core)))
    if N % core != 0:
        sum_result += diffusion_model.diffuse_times(seeds, N % core)

    pool.close()
    pool.join()

    for i in result:
        sum_result += i.get()

    print(sum_result / N)

    sys.stdout.flush()


if __name__ == '__main__':
    main()
