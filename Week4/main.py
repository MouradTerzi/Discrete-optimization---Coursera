from collections import namedtuple
from week4_assignment import *

Point = namedtuple("Point", ['x', 'y'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
    
    d = compute_euclidean_distance(nodeCount,points)
    tour,objective = greedy_heuristic(nodeCount,d)
    solution = construct_solution_from_tour(nodeCount,tour)
    print(f'Found tour:{solution}| Objective: {objective}')
    
    return 


if __name__ == '__main__':
    path = 'data/tsp_51_1'
    input_data = open(path,'r').read()
    solve_it(input_data)
    