import random
import copy
import time

def read_pmedian_instance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    number_vertices, number_edges, p = map(int, lines[0].split())
    distances = [[float('inf')] * number_vertices for _ in range(number_vertices)]

    for line in lines[1:]:
        i, j, k = map(int, line.split())
        distances[i-1][j-1] = k
        distances[j-1][i-1] = k

    for k in range(number_vertices):
        for i in range(number_vertices):
            for j in range(number_vertices):
                distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])

    return number_vertices, p, distances

# talvez de pra ser mais rapido se tivermos uma lista de vizinhos mais proximos por no. ai para identificar a nearest facility pegamos o no da solucao com menor index
def calculate_cost(solution, distances):
    total_cost = 0
    for client in [i for i in range(len(distances)) if i not in solution]:
        nearest_facility = min(solution, key=lambda facility: distances[client][facility])
        total_cost += distances[client][nearest_facility]
    return total_cost

def initial_solution(number_vertices, p, distances):
    initial_solution = []
    while len(initial_solution) < p:
        initial_solution, _ = V_add(initial_solution, [], number_vertices, distances)

    return initial_solution

def stopping_criteria(k, max_iterations):
    return k >= max_iterations

def V_add(Sc, tabu_list, number_vertices, distances):
    neighbours = []
    for node in range(number_vertices):
        if node not in tabu_list and node not in Sc:
            new_neighbour = copy.deepcopy(Sc)
            new_neighbour.append(node)
            neighbours.append(new_neighbour)

    best_neighbour = min(neighbours, key=lambda sol: calculate_cost(sol, distances))
    added_node = best_neighbour[-1]
    return best_neighbour, added_node

def V_drop(Sc, distances):
    min_cost = float("inf")
    for facility in Sc:
        drop_set = copy.deepcopy(Sc)
        drop_set.remove(facility)
        drop_cost = calculate_cost(drop_set, distances)
        if drop_cost < min_cost:
            min_cost = drop_cost
            dropped_facility = facility

    Sc.remove(dropped_facility)
    return Sc

def tabu_search(number_vertices, p, distances, max_iterations=100):
    S_0 = initial_solution(number_vertices, p, distances)
    S_star = copy.deepcopy(S_0)
    S_c = copy.deepcopy(S_0)
    tabu_list = []
    max_size_tabu = 10
    slack = 0
    stable_iterations = 0
    max_stable_iterations = 10
    sol_evolution = []
    sizesol_evolution = []

    k = 0
    while not stopping_criteria(k, max_iterations):
        #s_time = time.time()
        k += 1
        stable_iterations += 1
        if stable_iterations > max_stable_iterations:
            slack += 1
            stable_iterations = 0
        #print(k)

        if len(S_c) < p - slack:
            procedure = "add"
        elif len(S_c) > p + slack:
            procedure = "drop"
        else:
            coin = random.uniform(0, 1)
            if coin < 0.5 and len(S_c) > 2:
                procedure = "drop"
            else:
                procedure = "add"

        if procedure == "add":
            best_neighbour, added_node = V_add(S_c, tabu_list, number_vertices, distances)
            tabu_list.append(added_node)
            if len(tabu_list) > max_size_tabu:
                tabu_list.pop(0)
        else:
            best_neighbour = V_drop(S_c, distances)
        
        S_c = copy.deepcopy(best_neighbour)
        sizesol_evolution.append(len(S_c))

        #print(len(S_c))

        if len(S_c) == p and calculate_cost(S_c, distances) < calculate_cost(S_star, distances):
            S_star = copy.deepcopy(S_c)
            slack = 0
            stable_iterations = 0
            print(calculate_cost(S_star, distances))

        sol_evolution.append(calculate_cost(S_star, distances))
        #e_time = time.time()
        #print("it time", e_time - s_time)

    return S_star, calculate_cost(S_star, distances), sol_evolution, sizesol_evolution


def new_tabu_search(number_vertices, p, distances, max_iterations=100):
    random.seed(number_vertices * p)
    S_0 = initial_solution(number_vertices, p, distances)
    S_star = copy.deepcopy(S_0)
    S_c = copy.deepcopy(S_0)
    tabu_list = []
    max_size_tabu = 10
    sol_evolution = []
    new_sizesol_evolution_list = []

    k = 0
    while not stopping_criteria(k, max_iterations):
        k += 1
        #s_time = time.time()
        #print(k)

        if len(S_c) == p:
            best_neighbour, added_node = V_add(S_c, tabu_list, number_vertices, distances)
            tabu_list.append(added_node)
            if len(tabu_list) > max_size_tabu:
                tabu_list.pop(0)
        else:
            best_neighbour = V_drop(S_c, distances)
        
        S_c = copy.deepcopy(best_neighbour)

        new_sizesol_evolution_list.append(len(S_c))

        #print(len(S_c))

        if len(S_c) == p and calculate_cost(S_c, distances) < calculate_cost(S_star, distances):
            S_star = copy.deepcopy(S_c)
            print(calculate_cost(S_star, distances))

        sol_evolution.append(calculate_cost(S_star, distances))

        #e_time = time.time()
        #print("it time", e_time - s_time)

    return S_star, calculate_cost(S_star, distances), sol_evolution, new_sizesol_evolution_list


import os

path = 'content/instances/'
notebook = dict()
total_cost = []
new_total_cost = []
sol_evolution_list = []
new_sol_evolution_list = []
sizesol_evolution_list = []
new_sizesol_evolution_list = []
p_list = []
total_time = []
new_total_time = []

txt_files = [filename for filename in os.listdir(path) if filename.endswith('.txt')]
sorted_files = sorted(txt_files, key=lambda x: int(''.join(filter(str.isdigit, x))))


for filename in sorted_files:
    if filename.endswith('.txt'): # and (filename in ["pmed1.txt", "pmed2.txt", "pmed3.txt"]): # , "pmed4.txt", "pmed5.txt", "pmed6.txt", "pmed7.txt", "pmed8.txt", "pmed9.txt"]: or filename.startswith("pmed1")):
        file_path = os.path.join(path, filename)
        n, p, distances = read_pmedian_instance(file_path)
        p_list.append(p)

        start_time = time.time()
        best_solution, best_cost, sol_evolution, sizesol_evolution = tabu_search(n, p, distances)
        end_time = time.time()
        total_time.append(end_time - start_time)

        start_time = time.time()
        new_best_solution, new_best_cost, new_sol_evolution, new_sizesol_evolution = new_tabu_search(n, p, distances)
        end_time = time.time()
        new_total_time.append(end_time - start_time)

        total_cost.append(best_cost)
        new_total_cost.append(new_best_cost)
        sol_evolution_list.append(sol_evolution)
        new_sol_evolution_list.append(new_sol_evolution)
        sizesol_evolution_list.append(sizesol_evolution)
        new_sizesol_evolution_list.append(new_sizesol_evolution)

        end_time = time.time()

        # Print the results
        print(f"Instance: {filename}")
        print(f"Facilities: {best_solution}")
        print(f"Total Distance: {best_cost}")
        print("\n")

# grafico de valor encontrado por instancia
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
iterations = [i for i in range(len(total_cost))]
plt.plot(iterations, total_cost, label='Busca Tabu', marker='o')
plt.plot(iterations, new_total_cost, label='Busca Tabu Personalizada', marker='o')
optimal_values = [5819, 4093, 4250, 3034, 1355, 7824, 5631, 4445, 2734, 1255, 7696, 6634, 4374, 2968, 1729, 8162, 6999, 4809, 2845, 1789, 9138, 8579, 4619, 2961, 1828, 9917, 8307, 4498, 3033, 1989, 10086, 9297, 4700, 3013, 10400, 9934, 5057, 11060, 9423, 5128]
plt.plot(iterations, optimal_values, label='Otimo', marker='o')
plt.xlabel('instancia')
plt.ylabel('Valor da Solucao')
plt.title('Comparacao das solucoes')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('graficos/total.png')

# grafico de evolucao da solucao
for i in range(len(sol_evolution_list)):
    plt.figure(figsize=(10, 6))
    iterations = [i for i in range(len(sol_evolution_list[i]))]
    plt.plot(iterations, sol_evolution_list[i], label='Busca Tabu', marker='o')
    plt.plot(iterations, new_sol_evolution_list[i], label='Busca Tabu Personalizada', marker='o')
    plt.xlabel('iteracao')
    plt.ylabel('Valor da Solucao')
    plt.title('Comparacao das solucoes ' + str(i))
    plt.legend()
    plt.grid(True)
    #plt.show()
    figname = 'graficos/sol_pmed' + str(i+1) + '.png'
    plt.savefig(figname)


# grafico de tamanho da solucao corrente
for i in range(len(sizesol_evolution_list)):
    feasible_list = [p_list[i]] * len(sizesol_evolution_list[i])
    plt.figure(figsize=(10, 6))
    iterations = [i for i in range(len(sizesol_evolution_list[i]))]
    plt.plot(iterations, feasible_list, label = 'Max feasible size')
    plt.plot(iterations, sizesol_evolution_list[i], label='Busca Tabu')
    plt.plot(iterations, new_sizesol_evolution_list[i], label='Busca Tabu Personalizada')
    plt.xlabel('iteracao')
    plt.ylabel('Tamanho da Solucao')
    plt.title('Comparacao das solucoes ' + str(i))
    plt.legend()
    plt.grid(True)
    #plt.show()
    figname = 'graficos/len_pmed' + str(i+1) + '.png'
    plt.savefig(figname)


with open('table1.txt', 'w') as f:
    for i in range(len(total_cost)):
        f.write('pmed' + str(i+1) + " & " + str(optimal_values[i]) + " & " + str(total_cost[i]) + " & " + str(round(total_time[i], 1)) + " & " + str(new_total_cost[i]) + " & " + str(round(new_total_time[i], 1)) + " \\\ \n \hline \n") 

