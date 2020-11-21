from typing import List
from random import random

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
                    p: float = random()
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
            threshold.append(random())

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
