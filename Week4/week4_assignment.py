import numpy as np
from collections import Counter,defaultdict 
import time
import networkx as nx
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import math 

"""_summary_
    Helper methods
"""

def compute_euclidean_distance(nodeCount,points):
    d = list(np.zeros((nodeCount,nodeCount)))
    for i in tqdm(range(nodeCount)):
        for j in range(i+1,nodeCount):
            dij = pow(points[i][0] - points[j][0],2) + pow(points[i][1] - points[j][1],2)
            dij = round(math.sqrt(dij),2)
            d[i][j],d[j][i] = dij,dij

    return d

def construct_solution_from_tour(nodeCount,tour):
    solution = [tour[0][0],tour[0][1]]
    tour.remove(tour[0])
    current_vertex = solution[1]
    while(len(solution) < nodeCount):
        #print(len(solution))
        for (i,j) in tour:
            if i == current_vertex:
                solution.append(j)
                current_vertex = j
                tour.remove((i,j))
                break
            elif j == current_vertex:
                solution.append(i)
                current_vertex = i
                tour.remove((i,j))
                break

    return solution

def compute_objective(solution,d):
    objective = 0
    for i in range(len(solution)-1):
        objective+= d[solution[i]][solution[i+1]]

    objective += d[solution[len(solution)-1]][solution[0]]

    return objective

def save_solution(nodeCount,solution,objective,path,exec):

  f = open(path,'w')
  f.writelines(f'Total number of nodes:{nodeCount}\n')
  f.writelines(f'Solution:\n')
  s = ''.join(str(solution))
  f.writelines(f'{s}\n')
  f.writelines(f'Objective:{objective}\n')
  f.writelines(f'CPU time in seconds : {round(exec,2)}')

  return

def check_tour(tour,nodeCount):
    print(len(tour))
    occurences = dict(Counter(tour))
    [print(key,occurences[key]) for key in occurences]
    nodes_degree = []
    for i in range(nodeCount):
       degree_node = 0
       for e in tour:
           if e[0] == i or e[1] == i:
               degree_node +=1

       if degree_node == 2:
           nodes_degree.append([i,degree_node])

    print(nodes_degree)
    return

def construct_tour_from_solution(solution):

  tour = [(solution[i],solution[i+1]) for i in range(len(solution)-1)]
  tour.append((solution[len(solution)-1],solution[0]))
  return tour

"""_summary_
    Greedy heuristic
"""

def greedy_heuristic(nodeCount,d):
    tour = list()
    degree_nodes = defaultdict(int)
    nb_edges = 0
    start = time.time()
    with tqdm(total=nodeCount, desc="Processing") as pbar:
        while(nb_edges <= nodeCount):
            edges = [(i,j,d[i][j]) for i in range(nodeCount) for j in range(i+1,nodeCount) if ((i,j) not in tour) and (degree_nodes[i] != 2 and degree_nodes[j] != 2)]
            edges = sorted(edges,key=lambda x:x[2],reverse=False)
            add_new_edge(nodeCount,tour,degree_nodes,edges)
            nb_edges+=1
            pbar.update(1)

    end = time.time()
    print(f'Total CPU time : {end - start}')
    tour_for_objective = tour[:]
    solution = construct_solution_from_tour(nodeCount,tour_for_objective)

    #Construct the ordered tour and compute the objective value
    tour = construct_tour_from_solution(solution)
    objective = compute_objective(solution,d)

    return tour,objective

def add_new_edge(nodeCount,tour,degree_nodes,edges):
    index = 0
    while index < len(edges):
        i,j = edges[index][0],edges[index][1]

        #Check if there is a cycle when we insert (i,j)
        tour.append((i,j))
        degree_nodes[i]+=1
        degree_nodes[j]+=1
        G = nx.Graph()
        G.add_nodes_from([node for node in degree_nodes if degree_nodes[node] > 0])
        G.add_edges_from(tour)
        try:
            cycles = nx.find_cycle(G, orientation="ignore")
            if len(cycles) == nodeCount:
                return
            tour.pop()
            degree_nodes[i]-=1
            degree_nodes[j]-=1
            index+=1

        except nx.exception.NetworkXNoCycle:
            return

"""_summary_
    Nearest neighbor heuristic
"""

def nearest_neighbor_heuristic(nodeCount,d):
    current_node = 0
    solution = [current_node] #Select the first node
    while(len(solution)<nodeCount):
        d_current_node = [(j,d[current_node][j]) for j in range(nodeCount) if (j != current_node and j not in solution)]
        d_current_node = sorted(d_current_node,key = lambda x:x[1],reverse=False)
        current_node = d_current_node[0][0]
        solution.append(current_node)

    print(solution)

    return solution

"""_summary_
    Improvment heuristic (2-opt)
"""

def two_opt_improvment(nodeCount,tour,d,objective):

   improvment = True
   print('Initial objective:',objective)
   cpt = 1
   exec = 0
   start = time.time()
   while improvment == True and exec < 10:
       index = 0
       improvment = False
       while(index < len(tour)):
          index_improvment = False
          i,j = tour[index][0],tour[index][1]
          #print(f'Current index : {index}, Current edge: ({i,j})')
          edges = [(k,l) for (k,l) in tour if (k not in [i,j] and l not in [i,j] and (i,k) not in tour and (j,l) not in tour and (k,i) not in tour and (l,j) not in tour)]
          best_objective = objective
          #print(f'Selected edges:{edges}')
          for edge in edges:
              k,l = edge[0],edge[1]
              new_objective = objective - d[i][j] - d[k][l] + d[i][k] + d[j][l]
              if new_objective < best_objective:
                     best_objective = new_objective
                     move = k,l
                     index_improvment = True

          if index_improvment:
              print(f'{cpt} | Accepted 2-opt for ({i,j}) is ({move[0],move[1]})|New objective : {best_objective}')
              cpt+=1
              tour.remove((i,j))
              tour.remove((move[0],move[1]))
              tour.extend([(i,move[0]),(j,move[1])])
              #print(f'New tour: {tour}')
              objective = best_objective
              improvment = True
              solution = construct_solution_from_tour(nodeCount,tour)
              tour = construct_tour_from_solution(solution)
              #print(f'Ordered tour:{tour}')
              #index = 0

          else:
              index += 1

          exec = time.time() - start
          #print(' ')
   print(f'2-opt objective : {objective}')
   print(f'output tour: {tour}')
   print(f'CPU time in seconds: {round(exec,2)}')
   print(' ')
   return tour, objective,exec

def simulated_annealing(nodeCount,d,T0,Tf,rate,results_path):

    s,objective = greedy_heuristic(nodeCount,d)
    best_s = s[:]
    best_objective = objective
    T = T0
    it_list,best_it_list = [],[]
    s_list, best_s_list = [],[]
    it = 0
    start = time.time()
    exec = 0
    with tqdm(total=nodeCount, desc="Processing") as pbar:
        while(T >= Tf and exec < 17995):
          #1. Generate new solution from the neighborhood of the current one
          sv = s[:]
          sv,sv_objective = generate_candidate_solution(nodeCount,d,sv,objective)

          #2. Apply 2-opt improvment on the generate solution
          sv,sv_objective = two_opt_improvment(nodeCount,sv,d,sv_objective)

          #3. Accept or reject the new solution
          if sv_objective <= objective:
              s = sv[:]
              objective = sv_objective
          else:
              r = random.uniform(0,1)
              delta_obj = sv_objective - objective
              if r <= math.exp(-delta_obj/T):
                  s = sv[:]
                  objective = sv_objective

          it_list.append(it)
          s_list.append(objective)

          #4. Update the best solution yet found
          if objective <= best_objective:
              print(it,objective)
              best_s = s[:]
              best_objective = objective
              best_it_list.append(it)
              best_s_list.append(best_objective)

          T *= rate
          it+=1
          exec = time.time() - start
          pbar.update(1)

    solution = construct_solution_from_tour(nodeCount,best_s)
    create_figures(it_list,s_list,'Current solution per iteration',f'{results_path}_current_solution','b')
    create_figures(best_it_list,best_s_list,'Best solution per iteration',f'{results_path}_best_solution','r')
    save_solution(nodeCount,solution,best_objective,results_path)

    print(f'Total execution time : {round(exec,3)}')
    print(f'Best tour found : {solution}')
    print(f'Objective:{best_objective}')

    return solution

def generate_candidate_solution(nodeCount,d,input_tour,input_objective):

    i,j = random.choice(input_tour)

    input_tour.remove((i,j))
    edges = [(k,l) for (k,l) in input_tour if (k not in [i,j] and l not in [i,j] and (i,k)
             not in input_tour and (j,l) not in input_tour and (k,i) not in input_tour and (l,j) not in input_tour)]

    k,l = random.choice(edges)
    input_tour.remove((k,l))
    input_tour.extend([(i,k),(j,l)])
    new_objective = input_objective - d[i][j] - d[k][l] + d[i][k] + d[j][l]
    solution = construct_solution_from_tour(nodeCount,input_tour)
    sv_tour = construct_tour_from_solution(solution)

    return sv_tour,new_objective

def create_figures(x_axis,y_axis,title,figure_path,color):

  plt.scatter(x_axis,y_axis,color = color)
  plt.title(title)
  plt.savefig(figure_path)
  plt.clf()

  return

"""_summary_
    Resolution method
"""

def get_tour_greedy_heuristic(nodeCount,points,results_path):
    d = compute_euclidean_distance(nodeCount,points)
    tour,objective = greedy_heuristic(nodeCount,d)
    tour,objective = two_opt_improvment(nodeCount,tour,d,objective)
    print(tour)
    solution = construct_solution_from_tour(nodeCount,tour)
    print(solution)
    save_solution(nodeCount,solution,objective,results_path)

    return

def get_tour_nearest_neighbor(nodeCount,points,results_path):
    d = compute_euclidean_distance(nodeCount,points)
    solution = nearest_neighbor_heuristic(nodeCount,d)
    objective = compute_objective(solution,d)
    tour = construct_tour_from_solution(solution)
    tour,objective,exec = two_opt_improvment(nodeCount,tour,d,objective)
    #print(f'output tour: {tour}')
    solution = construct_solution_from_tour(nodeCount,tour)
    #print(solution)
    save_solution(nodeCount,solution,objective,results_path,exec)

    return