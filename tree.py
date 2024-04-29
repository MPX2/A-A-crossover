import math
import pandas as pd
import random 
import numpy as np

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
    initial_list2 = [0]


    coordinate = [] #경로들의 집합들을 모아놓은 리스트 즉  한 세대를 의미함. -> route로 이루어짐node안씀
    index_list = [] #방문한 인덱스로 이루어진 인덱스 
    data =[]#엑셀 데이터를 sort한후 저장된 리스트 -> x,y로 이루어짐


    def __init__(self,data, route_count): #매개변수로 route_count = 한세대당 개체개수와 genetic_count = 유전알고리즘을 몇세대 반복할건지 받음 근데 이건 
        ##data는 원점으로부터 짧은순서대로 오름차순 정렬되어있어야됨
        #한세대당 개수인 route_count의 수만큼 원점에서 가장 가까운 점 route_count개를 뽑아서 초기 경로를 route_count개를 만드는 과정
        self.route_count = route_count
        self.data = data
        for i in range(0,route_count):
            init_list = [[0, 0]]
            self.coordinate.append(init_list)
            self.index_list.append(self.initial_list2)

    

    def A_star(self): #에이스타 알고리즘으로 경로를 정하는 경로를 정하는 과정 
        for i in range(0,self.route_count):
            copy_array = [] #정렬된 데이터를노드화 해서 여기다 집어넣을것.
            for k in range(0,len(self.data)):
                new_Node = Node(self.data[k][0], self.data[k][1], k)  #copy_array는 노드형태로 리스트에 저장됨
                copy_array.append(new_Node)

            current_Node = copy_array[i]
            current_index = i #가장 짧은거 
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

data = pd.read_csv("2024_AI_TSP.csv") 
data_list = [] 
for i in range(0,len(data)):
    data_list.append([data.iat[i,0],data.iat[i,1]]) 

#원점으로부터 맨해튼거리의 합을 기준으로 오름차순 정렬
data_list.sort(key=lambda x: abs(x[1])+abs(x[0])) 

    
#에이스타 알고리즘 진행부분 
tree = A_star(data_list,5)
tree.A_star()
print(len(tree.coordinate[1]))

def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def cal_func(list): # 적합도 함수 -> 총 거리 합의 역수
    total_distance = 0
    for i in range(len(list) - 1):
        city1 = list[i]
        city2 = list[i + 1]
        total_distance += distance(city1, city2)

    print(total_distance)

list = np.array(tree.coordinate[2])
cal_func(list)
print("\n")




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

    



    
