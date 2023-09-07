import numpy as np 
from tqdm import tqdm 

def dynamic_programming(items,capacity,item_count):
    value = 0
    taken = [0]*len(items)
    capacity_items = np.zeros((capacity+1,item_count+1))
    for item in range(1,item_count+1):
        value_i = items[item-1][1]
        weight_i = items[item-1][2]
        for k in range(capacity+1):
            if weight_i <= k:
                capacity_items[k][item] = max(capacity_items[k][item-1],capacity_items[k-weight_i][item-1] + value_i)
            else:
                capacity_items[k][item] = capacity_items[k][item-1]
    
    item,k = item_count,capacity
    while item > 0:
        if capacity_items[k,item] != capacity_items[k,item-1]: #item is selected
            value_i = items[item-1][1]
            weight_i = items[item-1][2]
            taken[item-1] = 1
            k -= weight_i
            value += value_i
        
        item -= 1
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    
    return output_data

def greedy_heuristic(items,capacity):
    value = 0
    weight = 0
    taken = [0]*len(items)
    
    #Solve the problem using greedy heuristic
    untaken_items = [(item[0],item[1]/item[2]) for item in items]
    untaken_items = sorted(untaken_items,key=lambda x: x[1],reverse=True)
    
    for i in(range(len(untaken_items))):
        item_i = untaken_items[i][0]
        value_i = items[item_i][1]
        weight_i = items[item_i][2]
        if weight + weight_i <= capacity:
            taken[item_i] = 1
            weight+= weight_i
            value += value_i
            
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    
    return output_data 

def dynamic_programming_heuristic(items,capacity,item_count):
    value = 0
    weight = 0
    taken = [0]*len(items)
    max_weight = max([item[2] for item in items])
    
    #Solve the problem using heuristic basd on dynamic programming
    blocs = int(capacity/max_weight)
    for bloc in range(blocs):
        capacity_items = np.zeros((max_weight+1,item_count+1))
        for item in tqdm(range(1,item_count+1)):
            value_i = items[item-1][1]
            weight_i = items[item-1][2]
            for k in range(max_weight+1):
                if weight_i <= k:
                    capacity_items[k][item] = max(capacity_items[k][item-1],capacity_items[k-weight_i][item-1] + value_i)
                else:
                    capacity_items[k][item] = capacity_items[k][item-1]
        
        item,k = item_count,max_weight
        while item > 0:
            if capacity_items[k,item] != capacity_items[k,item-1]: #item is selected
                index_i = items[item-1][0]
                value_i = items[item-1][1]
                weight_i = items[item-1][2]
                taken[index_i] = 1
                item_count -= 1
                weight += weight_i
                del items[item-1]
                k -= weight_i
                value += value_i
        
            item -= 1
    remaining_weight = capacity - weight
    capacity_items = np.zeros((remaining_weight+1,item_count+1))
    for item in tqdm(range(1,item_count+1)):
            value_i = items[item-1][1]
            weight_i = items[item-1][2]
            for k in range(remaining_weight+1):
                if weight_i <= k:
                    capacity_items[k][item] = max(capacity_items[k][item-1],capacity_items[k-weight_i][item-1] + value_i)
                else:
                    capacity_items[k][item] = capacity_items[k][item-1]
    
    item,k = item_count,remaining_weight
   
    while item > 0:
        if capacity_items[k,item] != capacity_items[k,item-1]: #item is selected
            index_i = items[item-1][0]
            value_i = items[item-1][1]
            weight_i = items[item-1][2]
            taken[index_i] = 1
            weight += weight_i
            k -= weight_i
            value += value_i
    
        item -= 1
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    
    return output_data 