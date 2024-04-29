import pandas as pd
import numpy as np
from random import shuffle
import random
import time
import math

start_time = time.time()

#여러 도시들의 경로들을 모아둔 route클래스 구현 

class route:
   
    def __init__(self): 
        self.city_route = [] #초기 경로 리스트에 0,0을 삽입해주면서 초기화함
        self.city_route.append([0,0])
    
    def __getitem__(self, index):
        return self.city_route[index]

    def insert(self, node): #좌표를 매개변수로 받으면 그걸 리스트에 추가하는 함수 
        self.city_route.append(node) #입력받은 노드를 경로에 추가
    
    
    def get_terminal(self): #마지막 노드 반환
        return self.city_route[-1]
    
    def calmulate_route_length(self):#0,0에서부터 현재 노드까지의 누적 거리의 합을 반환하는 함수
        length = 0
        for i in range(0,len(self.city_route)-1):
            pre_x = self.city_route[i][0]
            pre_y = self.city_route[i][1]
            nex_x = self.city_route[i+1][0]
            nex_y = self.city_route[i+1][1]
            length += math.sqrt((pre_x - nex_x)**2 + (pre_y - nex_y)**2)
        length += math.sqrt((self.city_route[len(self.city_route)-1].get_x())**2 + (self.city_route[len(self.city_route)-1].get_y())**2)

        return length


 #####################################################################          

#도시하나를 의미하는 Node구현 노드 하나에는 도시의 x좌표,y좌표 그리고 sorted된 데이터집합에서 도시의 인덱스 번호가 들어가 있음 
class Node:
    def __init__(self, x, y,index):
        self.x = x
        self.y = y
        self.index = index
    
    def get_heuristic(self): #현재 노드에서의 휴리스틱함수를 
        test = self.manhattan_distance()
        if test == 0:
            heuristic_value = 0
        else:
            heuristic_value = 1/ self.manhattan_distance()
        return heuristic_value
    #구현하기
    def add_child(self, child): #하나의끝 좌표에 자식 
        self.children.append(child)

    def uclid_distance(self, node): #해당 노드까지 유클리드 거리반환
        return math.sqrt(abs(self.x - node.x)**2 + abs(self.y - node.y)**2)

    def manhattan_distance(self): #노드의 맨해튼 거리 반환
        return abs(self.x) + abs(self.y)
    def get_index(self):
        return self.index
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y

    
#에이스타 알고리즘 구현부분 

class A_star: #맨처음 원점에서 가장 짧은 것들을 선택하고 그 이후 에이스타 알고리즘 서치를 통해서 경로를 구성하는 트리 


    coordinate = [] #경로들의 집합들을 모아놓은 리스트 즉  한 세대를 의미함. -> route로 이루어짐node안씀
    index_list = [] #방문한 인덱스로 이루어진 인덱스 
    data =[]#엑셀 데이터를 sort한후 저장된 리스트 -> x,y로 이루어짐


    def __init__(self,data, route_count): #매개변수로 route_count = 한세대당 개체개수와 genetic_count = 유전알고리즘을 몇세대 반복할건지 받음 근데 이건 
        ##data는 원점으로부터 짧은순서대로 오름차순 정렬되어있어야됨
        #한세대당 개수인 route_count의 수만큼 원점에서 가장 가까운 점 route_count개를 뽑아서 초기 경로를 route_count개를 만드는 과정
        self.route_count = route_count
        self.data = data
        for i in range(0,route_count):
            init_list = []
            init_list2 = []
            self.coordinate.append(init_list)
            self.index_list.append(init_list2)
    

    def A_star(self): #에이스타 알고리즘으로 경로를 정하는 경로를 정하는 과정 
        for i in range(0,self.route_count):
            copy_array = [] #정렬된 데이터를노드화 해서 여기다 집어넣을것.
            self.coordinate[i].append([0,0])
            self.index_list[i].append(0)

            for k in range(0,len(self.data)):
                new_Node = Node(self.data[k][0], self.data[k][1], k)  #copy_array는 노드형태로 리스트에 저장됨
                copy_array.append(new_Node)

            copy_array.pop(0)
            current_Node = copy_array[i+1]
            current_index = i+1 #가장 짧은거 
            while len(copy_array) > 1:
                max = 0
                next_index = -1
                if current_index-50<0 and current_index+50>len(copy_array)-1:
                    for k in range(0, len(copy_array)):
                        if k == current_index:
                            continue
                        p_p_list = current_Node.uclid_distance(copy_array[k]) #실제 다음 후보노드까지의 거리계산                       
                        heuristic_value = copy_array[k].get_heuristic()#휴리스틱계산
                        fn_value = 2*p_p_list + heuristic_value #f(N)계산
                        if fn_value > max:
                            next_index = k
                            max = fn_value
                    self.index_list[i].append(copy_array[current_index].get_index())
                    self.coordinate[i].append([copy_array[current_index].get_x(), copy_array[current_index].get_y()])
                    current_Node = copy_array[next_index]
                    copy_array.pop(current_index)
                    current_index = next_index
                    if current_index != 0:
                        current_index -= 1
                    
                elif current_index-50<0:
                    for k in range(0, current_index+50):
                        if k == current_index:
                            continue
                        p_p_list = current_Node.uclid_distance(copy_array[k]) #실제 다음 후보노드까지의 거리계산                       
                        heuristic_value = copy_array[k].get_heuristic()#휴리스틱계산
                        fn_value = 2*p_p_list + heuristic_value #f(N)계산
                        if fn_value > max:
                            next_index = k
                            max = fn_value
                    self.index_list[i].append(copy_array[current_index].get_index())
                    self.coordinate[i].append([copy_array[current_index].get_x(), copy_array[current_index].get_y()])
                    current_Node = copy_array[next_index]
                    copy_array.pop(current_index)
                    current_index = next_index
                    if current_index != 0:
                        current_index -= 1

                elif current_index+50>len(copy_array)-1:
                    for k in range(current_index-50, len(copy_array)):
                        if k == current_index:
                            continue
                        p_p_list = current_Node.uclid_distance(copy_array[k]) #실제 다음 후보노드까지의 거리계산                       
                        heuristic_value = copy_array[k].get_heuristic()#휴리스틱계산
                        fn_value = 2*p_p_list + heuristic_value #f(N)계산
                        if fn_value > max:
                            next_index = k
                            max = fn_value
                    self.index_list[i].append(copy_array[current_index].get_index())
                    self.coordinate[i].append([copy_array[current_index].get_x(), copy_array[current_index].get_y()])
                    current_Node = copy_array[next_index]
                    copy_array.pop(current_index)
                    current_index = next_index
                    if current_index != 0:
                        current_index -= 1
        
                else:
                    for k in range(current_index-50, current_index+50):
                        if k == current_index:
                            continue
                        p_p_list = current_Node.uclid_distance(copy_array[k]) #실제 다음 후보노드까지의 거리계산                       
                        heuristic_value = copy_array[k].get_heuristic()#휴리스틱계산
                        fn_value = 2*p_p_list + heuristic_value #f(N)계산
                        if fn_value > max:
                            next_index = k
                            max = fn_value
                    self.index_list[i].append(copy_array[current_index].get_index())
                    self.coordinate[i].append([copy_array[current_index].get_x(), copy_array[current_index].get_y()])
                    current_Node = copy_array[next_index]
                    copy_array.pop(current_index)
                    current_index = next_index
                    if current_index != 0:
                        current_index -= 1


            self.index_list[i].append(copy_array[-1].get_index())
            self.coordinate[i].append([copy_array[-1].get_x(), copy_array[current_index].get_y()])
                

    
#해의 적합도를 판단하는 fitness 함수구현 
#매개변수로 route 즉 경로 하나가 주어짐.
    def fitness(self, route):
        distance = route.calmulate_route_length()
        index_gap = 0

        for i in range(1, 998):
            if abs(i - route[i].get_index()) < 10:
                index_gap += 20
            elif abs(i - route[i].get_index()) < 20:
                index_gap += 5
            else:
                continue

        return 50 / distance + 1 / index_gap #임의로 50과 1로 인덱스와 거리의 가중치를나눠둠 이거는 실제로 돌려보면서 최적의 값을 찾아야됨 
    



###########################################################################
    
#데이터로드하고 리스트에 저장

data = pd.read_csv("2024_AI_TSP.csv", header=None) 
data_list = [] 
for i in range(0,len(data)):
    data_list.append([data.iat[i,0],data.iat[i,1]]) 

#원점으로부터 맨해튼거리의 합을 기준으로 오름차순 정렬
data_list.sort(key=lambda x: abs(x[1])+abs(x[0])) 

    
#에이스타 알고리즘 진행부분 
tree = A_star(data_list, 50)
tree.A_star()
# print(len(tree.coordinate[1]))

# def distance(city1, city2):
#     return np.linalg.norm(city1 - city2)

# def cal_func(list): # 적합도 함수 -> 총 거리 합의 역수
#     total_distance = 0
#     for i in range(len(list) - 1):
#         city1 = list[i]
#         city2 = list[i + 1]
#         total_distance += distance(city1, city2)

#     print(total_distance)

# list = np.array(tree.coordinate[2])
# cal_func(list) # tree.index_list ## list == population
# print("\n") # datalist

# print(len(tree.index_list[49]))
# print(tree.index_list[49])



#방문한 인덱스번호를 순서대로 저장한 배열 = index_list





#크로스오버 + 에이스타
def a_star_crossover(route1, route2,num): #임의의 난수 뽑아서 인덱스 순으로 정렬 route1,route2는 경로2개이고 num은 997개의 점중에 몇개를 바꿀건지정하는 정수
    
    #나중에 매개변수에 depth추가해서 깊이마다 바뀌는 값 적어지게 하기.
    sort_list = [] #sort할때 이용할 리스트 
    numbers = np.random.choice(range(1, 997), num, replace=False)
    numbers.sort()
    for i in range(0,len(numbers)):
        sort_list.append(route1[numbers[i]])
    
    sort_list.sort(key = lambda x :x.get_index())
    for i in range(0,len(numbers)):
        index = numbers[i]
        route1.city_route[index] = sort_list[i]
 
    sort_list = []
    index_num = random.randint(1,997-num)
    for i in range(index_num, index_num+num):
        sort_list.append(route1.city_route[i])
    
    sort_list.sort(key = lambda x :x.get_index())

    for i in range(0,num):
        route1.city_route[i] = sort_list[i]
    
#에이스타 크로스오버 두가지 구현했는데 
#첫번째 주석처리되어있는건 997개점 중에 임의로 num개를뽑아서 그 점들을 인덱스순으로 정렬하는것->전체적인 리스트의 순서를 최적화 해줌 
#두번째 주석처리 안되어 있는건 임의의점 하나를 골라서 거기서부터 num개의 점들을 인덱스 순으로 정렬하는것 -> local에서 순서를 최적화 해줌 


#크로스오버 + 에이스타
def a_star_crossover(route1, route2,num): #임의의 난수 뽑아서 인덱스 순으로 정렬 route1,route2는 경로2개이고 num은 997개의 점중에 몇개를 바꿀건지정하는 정수
    
    #나중에 매개변수에 depth추가해서 깊이마다 바뀌는 값 적어지게 하기.
    sort_list = [] #sort할때 이용할 리스트 
    numbers = np.random.choice(range(1, 997), num, replace=False)
    numbers.sort()
    for i in range(0,len(numbers)):
        sort_list.append(route1[numbers[i]])
    
    sort_list.sort(key = lambda x :x.get_index())
    for i in range(0,len(numbers)):
        index = numbers[i]
        route1.city_route[index] = sort_list[i]
 
    sort_list = []
    index_num = random.randint(1,997-num)
    for i in range(index_num, index_num+num):
        sort_list.append(route1.city_route[i])
    
    sort_list.sort(key = lambda x :x.get_index())

    for i in range(0,num):
        route1.city_route[i] = sort_list[i]
    
#에이스타 크로스오버 두가지 구현했는데 
#첫번째 주석처리되어있는건 997개점 중에 임의로 num개를뽑아서 그 점들을 인덱스순으로 정렬하는것->전체적인 리스트의 순서를 최적화 해줌 
#두번째 주석처리 안되어 있는건 임의의점 하나를 골라서 거기서부터 num개의 점들을 인덱스 순으로 정렬하는것 -> local에서 순서를 최적화 해줌 

# CSV 파일에서 좌표 데이터를 가져옴
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    coordinates = df.values
    return coordinates


################ 초기 해 생성 ################
def initial_population(coordinates, population_size):
    population = []
    for _ in range(population_size):
        path = list(range(1, len(coordinates)))  # 첫 번째 요소를 제외한 경로 생성
        shuffle(path)
        path.insert(0, 0)
        population.append(path)
    return population

################ Evaluation Function ################

# 좌표 간 거리 계산
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def fitness(population, coordinates): # 적합도 함수 -> 총 거리 합의 역수
    fitness_values = []
    for individual in population:
        total_distance = 0
        for i in range(len(individual) - 1):
            city1 = coordinates[individual[i]]
            city2 = coordinates[individual[i + 1]]
            total_distance += distance(city1, city2)
        total_distance += distance(coordinates[individual[-1]], coordinates[individual[0]])
        fitness_values.append(1 / total_distance)  # 거리가 짧을수록 더 높은 적합도를 갖도록 역수를 취합니다.
    return fitness_values

def distance_calculate(population, coordinates): # 적합도 함수 -> 총 거리 합의 역수
    distance_values = []
    for individual in population:
        total_distance = 0
        for i in range(len(individual) - 1):
            city1 = coordinates[individual[i]]
            city2 = coordinates[individual[i + 1]]
            total_distance += distance(city1, city2)
        total_distance += distance(coordinates[individual[-1]], coordinates[individual[0]])
        distance_values.append(total_distance)
    return sum(distance_values) / len(distance_values)

################ Selection ################
def selection(population, fitness_values):
    # return roulette_wheel(population, fitness_values)
    return tournament_selection(population, fitness_values)
    # return rank_selection(population, fitness_values)
    return elitism_selection(population, fitness_values)

# def roulette_wheel(population, fitness_values):
#     total_fitness = sum(fitness_values)
#     probabilities = [fitness / total_fitness for fitness in fitness_values]
#     selected_index = np.random.choice(len(population), p=probabilities)
#     return population[selected_index]

def tournament_selection(population, fitness_values):
    tournament_size = 5
    selection_pressure = 0.9

    tournament = random.sample(range(len(population)), tournament_size)  # 토너먼트 크기만큼 무작위로 개체를 선택
    tournament_fitness = [fitness_values[i] for i in tournament]
    if random.random() < selection_pressure:
        tournament_winner = tournament[np.argmax(tournament_fitness)]  # 토너먼트에서 가장 적합도가 높은 개체의 인덱스 선택
    else:
        tournament_winner = tournament[np.argmin(tournament_fitness)]  # 토너먼트에서 가장 적합도가 낮은 개체의 인덱스 선택
    return population[tournament_winner]

# def rank_selection(population, fitness_values):
#     ranked_population = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k], reverse=True) # 순위 결정
#     selection_probabilities = [i / sum(range(1, len(population) + 1)) for i in range(1, len(population) + 1)] # 순위에 따른 확률 계산

#     selected_index = np.random.choice(ranked_population, p=selection_probabilities) # 순위에 따른 선택
#     return population[selected_index]

# def elitism_selection(population, fitness_values):

#     elite_indices = np.argmax(fitness_values)
#     return population[elite_indices]
    

################ Crossover ################
def crossover(parent1, parent2, crossover_rate):
    return singlepoint_crossover(parent1, parent2, crossover_rate)
    # return twopoint_crossover(parent1, parent2, crossover_rate)
    # return uniform_crossover(parent1, parent2, crossover_rate)
    # return er_crossover(parent1,parent2,crossover_rate)
    return cycle_crossover(parent1,parent2,crossover_rate)


def singlepoint_crossover(parent1, parent2, crossover_rate):

    def dup_indices(arr1, arr2):
        duplicate_indices = []
        
        # 배열2의 각 값에 대해 반복
        for index in range(len(arr2)):
            value = arr2[index]
            
            if value in arr1:
                duplicate_indices.append(index)
        
        return duplicate_indices
    
    def find_missingValues(array):
        # 집합으로 변환
        unique_values = set(array)

        complete_set = set(range(998))
        missing_values = sorted(complete_set - unique_values)
        
        return missing_values
    
    def singlepoint_func(parent1, parent2):
        crossover_point = random.randint(10, len(parent1) - 10)
        temparr = parent2[crossover_point:]
        duplicate_indices = dup_indices(parent1[:crossover_point], temparr)
        incomplete_arr = parent1[:crossover_point] + parent2[crossover_point:]
        missing_values = find_missingValues(incomplete_arr)
        if len(duplicate_indices) == len(missing_values):
            for i in range(len(duplicate_indices)):
                temparr[duplicate_indices[i]] = missing_values[i]
        else:
            print("len(duplicate_indices) =/= len(missing_values)!")
            exit(1)
        return parent1[:crossover_point] + temparr

    if random.random() < crossover_rate:
        child1 = singlepoint_func(parent1, parent2)
        child2 = singlepoint_func(parent2, parent1)
        return child1, child2
    else:
        return parent1, parent2

def er_crossover(parent1,parent2,crossover_rate):
    if random.random() < crossover_rate:
        num_cities = len(parent1)
        child1 = [None] * num_cities
        child2 = [None] * num_cities
        edges1 = {i: set() for i in range(num_cities+1)}
        edges2 = {i: set() for i in range(num_cities+1)}
        def add_edge(parent): #인접도시 목록 만들기
          for i in range(num_cities):
            left=parent[(i-1)%num_cities]
            right=parent[(i+1)%num_cities]
            edges1[parent[i]].update([left,right])
            edges2[parent[i]].update([left,right])
        add_edge(parent1)
        add_edge(parent2)
        
        current = parent1[0]
        child1 = [current]
        while len(child1) < len(parent1): 
            for edges in edges1.values():
                edges.discard(current) #방문한 도시는 인접도시 목록에서 삭제
            if edges1[current]:
                next_city = min(edges1[current], key=lambda x: len(edges1[x])) #현재 노드에서 인접 도시가 있다면 그중 남은 인접 도시가 가장 적은 지역 우선 방문
            else:
                remaining = set(parent1) - set(child1)
                next_city = random.choice(list(remaining)) #현재 노드에서 인접 도시가 없다면 전체에서 랜덤 방문
            child1.append(next_city)
            current = next_city

        current = parent2[0]
        child2 = [current]
        while len(child2) < len(parent2):
            for edges in edges2.values():
                edges.discard(current)
            if edges2[current]:
                next_city = min(edges2[current], key=lambda x: len(edges2[x]))
            else:
                remaining = set(parent2) - set(child2)
                next_city = random.choice(list(remaining))
            child2.append(next_city)
            current = next_city

        return child1,child2
    else:
        return parent1,parent2
    
def cycle_crossover(parent1,parent2,crossover_rate):
    if random.random() < crossover_rate:
        child1 = [-1] * len(parent1)
        child2 = [-1] * len(parent1)
        cycle_num = 0
        visited = [False] * len(parent1)

        while False in visited:
            if cycle_num % 2 == 0:
                dest1 = child1
                dest2 = child2
            else:  
                dest1 = child2
                dest2 = child1

            for start in range(len(parent1)):
                if not visited[start]:
                    break
            current = start
            while True:
                dest1[current] = parent1[current]
                dest2[current] = parent2[current]
                visited[current] = True
                current = parent1.index(parent2[current])
                if current == start:
                    break
            cycle_num += 1
        return child1, child2
    
    else:
        return parent1, parent2


# def twopoint_crossover(parent1, parent2, crossover_rate):
#     if random.random() < crossover_rate:
#         crossover_point1, crossover_point2 = 0, 0
#         while crossover_point1 == crossover_point2:
#             crossover_point1 = random.randint(0, len(parent1) - 1)
#             crossover_point2 = random.randint(0, len(parent1) - 1)
#             child1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[crossover_point2:]
#             child2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[crossover_point2:]
#         return child1, child2
#     else:
#         return parent1, parent2

# def uniform_crossover(parent1, parent2, crossover_rate):
#     if random.random() < crossover_rate:
#         child1 = [None] * len(parent1)
#         child2 = [None] * len(parent2)
        
#         for i in range(len(parent1)):
#             if random.random() < 0.5:
#                 child1[i] = parent1[i]
#                 child2[i] = parent2[i]
#             else:
#                 child1[i] = parent2[i]
#                 child2[i] = parent1[i]
        
#         return child1, child2
#     else:
#         return parent1, parent2

################ Mutation ################
def mutation(individual, mutation_rate):
    return equal_mutation(individual, mutation_rate)

def equal_mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(1, len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def notequal_mutation(individual, cur_generation):
    # # 1개의 유전자 선택 후 서로 교환
    # if random.random() < 0.1 / math.log10(10 + cur_generation):
    #     idx1, idx2 = random.sample(range(1, len(individual)), 2)
    #     individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    # return individual

    # 10개의 유전자를 선택해서 서로 교환
    if random.random() < 0.1 / math.log10(10 + cur_generation):
        indices = random.sample(range(1, len(individual)), 10)
        for i in range(0, len(indices), 2):
            idx1, idx2 = indices[i], indices[i+1]
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    
    return individual
   

################ GA ################
def genetic_algorithm(coordinates, population_size, generations, crossover_rate, mutation_rate, a_population):
    population = a_population
    best_distance = float('inf')
    best_path = None
    end_counter = 0
    genAvgDistance = []

    for cur_generation in range(generations):
        fitness_values = fitness(population, coordinates)
        new_population = []
        genAvgDistance.append(distance_calculate(population, coordinates))
        if cur_generation > 500:
            if abs(genAvgDistance[cur_generation - 1] - genAvgDistance[cur_generation]) < 0.5:
                end_counter += 1
            else:
                end_counter = 0
                
            if end_counter > 5:
                print("Program ended at generation %d" % (cur_generation + 1))
                return best_path
        for _ in range(population_size // 2):
            parent1 = selection(population, fitness_values)
            parent2 = selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = notequal_mutation(child1, cur_generation)
            child2 = notequal_mutation(child2, cur_generation) # 교차 과정으로 2개의 자식이 생성
            new_population.extend([child1, child2])

        population = new_population
        min_distance = 1 / max(fitness_values)  # 최소 거리는 최대 적합도의 역수
        if min_distance < best_distance:
            best_distance = min_distance
            best_path = population[np.argmax(fitness_values)]
        print("current generation : %d" % (cur_generation + 1))

    print("Program ended at generation %d" % (cur_generation + 1))
    return best_path

# 데이터 로드
coordinates = load_data("2024_AI_TSP.csv")

# 설정
population_size = 50 # 개체군의 크기(초기에 생성되는 경로의 개수)
generations = 50 # 수행할 반복 횟수
crossover_rate = 0.8
mutation_rate = 0.05 # 다양성 증가, local optimal 방지

# 유전 알고리즘 실행
best_path = genetic_algorithm(np.array(data_list), population_size, generations, crossover_rate, mutation_rate, tree.index_list)

# 최적 경로 저장
best_path_df = pd.DataFrame(best_path)
best_path_df.to_csv("best_path.csv", index=False, header=False)
print("\n")
print("best_path.csv' 저장 성공")
end_time = time.time()

# 실행 시간 계산
execution_time = end_time - start_time
print("실행시간 :", execution_time, "초")

# 실제 거리 평가
import TSP_eval