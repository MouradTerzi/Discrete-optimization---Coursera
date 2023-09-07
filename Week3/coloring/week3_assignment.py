import networkx as nx 
from ortools.sat.python import cp_model

"""
   # Degree of nodes and chromatic number
"""
def is_complete(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    max_possible_edges = num_nodes * (num_nodes - 1) / 2  # For an undirected graph

    return num_edges == max_possible_edges

def get_node_degree(n,edges):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(n)])
    G.add_edges_from(edges)
    G_is_complete = is_complete(G)
    
    max_degrees = n-1
    if not G_is_complete:        
        degrees = G.degree()
        degrees = sorted(degrees,key = lambda x: x[1],reverse = True)
        orderded_nodes = [x[0] for x in degrees]
        max_degrees = degrees[0][1]    
    
    return G_is_complete,max_degrees,orderded_nodes
    
"""
   # Find the maximum click
"""
def get_max_click(n,edges):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(n)])
    G.add_edges_from(edges)
    cliques = list(nx.find_cliques(G))
    max_clique = max(cliques,key = len)
    
    return len(max_clique)

def graph_coloring_constraint_programming(node_count,edge_count,edges,max_degrees,ordered_nodes):
    
    #Create the CpModel and initialize variables 
    model = cp_model.CpModel()
    colors = [model.NewIntVar(0, i, f"c_{i}") for i in range(node_count)]
    
    #Feasibility constraints 
    [model.Add(colors[i] != colors[j]) for (i,j) in edges]
    
    #Symmetry constraints 
    model.Add(colors[0] == 0)
    for i in range(1,node_count):
        max_ci = model.NewIntVar(0,i,"max_c{i}")
        model.AddMaxEquality(max_ci,colors[:i])
        model.Add(colors[i]<= max_ci+1)
    
    #Add the objective function 
    max_clique = get_max_click(node_count,edges)
    max_colors = model.NewIntVar(max_clique-1,max_degrees-1,"max_color")
    model.AddMaxEquality(max_colors,colors)
    model.Minimize(max_colors)
    
    #Solve the model, timelimit = 5 hours
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 18000

    print(f'resolution of the instance with {node_count} nodes')
    status = solver.Solve(model)
    
    #Show the running time 
    print("Time used (seconds):", solver.WallTime())
    
    if status == cp_model.OPTIMAL:
        solution = [solver.Value(colors[i]) for i in range(len(colors))]
        return solution, int(solver.ObjectiveValue()+1),1
        
    elif status == cp_model.FEASIBLE:
        solution = [solver.Value(colors[i]) for i in range(len(colors))]
        return solution, int(solver.ObjectiveValue()+1),0
        
    elif status == cp_model.INFEASIBLE:
        print("The model is infeasible.")
        return [],-1,0
        
    else:
        print("No solution found.")
        return [],-1,0

def graph_coloring(node_count,edge_count,edges): 
    """
      # Graph coloring assignment - Week 3
    """
    
    #Check if the graph is complete and get the max degree 
    graph_is_complete,max_degrees,ordered_nodes = get_node_degree(node_count,edges)
    if graph_is_complete : 
        return [i for i in range(node_count)],node_count,1
    else:
        solution,objective,optimal = graph_coloring_constraint_programming(node_count,edge_count,
                                                                           edges,max_degrees,
                                                                           ordered_nodes)
        return solution,objective,optimal
 