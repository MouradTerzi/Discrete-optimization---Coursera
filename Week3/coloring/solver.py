#!/usr/bin/python
# -*- coding: utf-8 -*-
from week3_assignment import *

def show_graph(node_count,edge_count,edges):
    print(f'Nodes : {node_count}')
    print(f'Edges : {edge_count}')
    return 

def write_problem(node_count,edge_count,edges):
    data = open(f'gc_{node_count}_1_assignment','w')
    data.writelines(f'{node_count} {edge_count}\n')
    [data.writelines(f'{i} {j}\n' for i,j in edges)]
    data.close()
    
    return
    
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    print(node_count)
    
    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
    
    #Resolution of the graph coloring problem
    solution, objective, optimal = graph_coloring(node_count,edge_count,edges)

    # prepare the solution in the specified output format
    output_data = str(objective) + ' ' + str(optimal) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

