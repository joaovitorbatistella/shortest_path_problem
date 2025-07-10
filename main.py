import os
import time
import heapq
import tracemalloc
import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from datetime import datetime

dirname = os.path.dirname(__file__)
base_path = f"{dirname}/output"

class DijkstraGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
    
    def add_vertex(self, name):
        self.vertices.add(name)
        if name not in self.graph:
            self.graph[name] = []
    
    def add_edge(self, origem, destino, peso):
        self.graph[origem].append((destino, peso))
        self.graph[destino].append((origem, peso))  # Grafo não direcionado
        self.vertices.add(origem)
        self.vertices.add(destino)
    
    def dijkstra(self, start, end=None):
        """
        Implements Dijkstra's algorithm
        Returns distances, predecessors, and the path (if end is specified)
        """
        # Iniciar medição de memória
        tracemalloc.start()
        start_time = time.perf_counter()
        comparisons = 0
        iterations = 0

        # Inicialização
        distances = {vertex: float('infinity') for vertex in self.graph} # O(1)
        distances[start] = 0
        predecessors = {vertex: None for vertex in self.graph} # O(1)
        visited = set()
        
        # Fila de prioridade (distância, vértice)
        pq = [(0, start)] # O(1)
        
        while pq: # O(V) iterações
            iterations += 1
            current_distance, current_vertex = heapq.heappop(pq)
            
            comparisons += 1
            if current_vertex in visited:  # O(1)
                continue
                
            visited.add(current_vertex) # O(1)
            
            # Se chegamos ao destino, podemos parar
            comparisons += 1
            if end and current_vertex == end: # O(1)
                break
            
            # RELAXAMENTO - Complexidade total: O(E * log V)
            # Grafo representado como lista de adjacências
            # grafo = {
            #     'A': ['B', 'C', 'D'],           # grau(A) = 3
            #     'B': ['A', 'C'],                # grau(B) = 2  
            #     'C': ['A', 'B', 'D', 'E'],      # grau(C) = 4
            #     'D': ['A', 'C'],                # grau(D) = 2
            #     'E': ['C']                      # grau(E) = 1
            # }
            for neighbor, weight in self.graph[current_vertex]: # O(grau(v))
                iterations += 1 # O(1)

                comparisons += 1
                if neighbor not in visited: # O(1)
                    new_distance = current_distance + weight
                    
                    comparisons += 1
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        predecessors[neighbor] = current_vertex
                        heapq.heappush(pq, (new_distance, neighbor)) # O(log V)
        
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        execution_time = (end_time - start_time) * 1000 # milliseconds
        memory_usage = peak / 1024 # KB

        tracemalloc.stop()

        metrics = {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'comparisons': comparisons,
            'iterations': iterations,
        }

        # Constrói o caminho se um destino foi especificado
        path = []
        if end and end in predecessors:
            current = end
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
        
        return distances, predecessors, path if end else None, metrics

class FloydWarshallGraph:
    def __init__(self):
        self.graph = defaultdict(dict)
        self.vertices = []
        self.distance_matrix = None
        self.next_matrix = None
    
    def add_vertices(self, vertices_list):
        self.vertices = vertices_list
        for vertex in vertices_list:
            self.graph[vertex] = {}
    
    def add_edge(self, origem, destino, peso):
        self.graph[origem][destino] = peso
        self.graph[destino][origem] = peso  # Grafo não direcionado
    
    def floyd_warshall(self):
        """
        Implementa o algoritmo de Floyd-Warshall
        Retorna matriz de distâncias e matriz de próximos vértices
        """

        # Iniciar medição de memória
        tracemalloc.start()
        start_time = time.perf_counter()
        comparisons = 0
        iterations = 0

        n = len(self.vertices)
        INF = float('inf')
        
        # Inicialização O(V²)
        dist = [[INF] * n for _ in range(n)]
        
        # Mapeamento O(V)
        vertex_to_index = {v: i for i, v in enumerate(self.vertices)}
        
        # Diagonal O(V)
        for i in range(n):
            iterations += 1
            dist[i][i] = 0
        
        # Arestas O(E)
        for u in self.graph:
            for v in self.graph[u]:
                iterations += 1
                i = vertex_to_index[u]
                j = vertex_to_index[v]
                dist[i][j] = self.graph[u][v]

        # Algoritmo principal de Floyd-Warshall -> O(V³) com otimização
        for k in range(n):  # Vértice intermediário
            for i in range(n):  # Vértice origem
                for j in range(n):  # Vértice destino
                    iterations += 1

                    comparisons +=1
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        execution_time = (end_time - start_time) * 1000 # milliseconds
        memory_usage = peak / 1024 # KB

        tracemalloc.stop()

        metrics = {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'comparisons': comparisons,
            'iterations': iterations,
        }
        
        # Detecta ciclos negativos
        for i in range(n):
            if dist[i][i] < 0:
                raise ValueError("Grafo contém ciclo negativo!")
        
        self.distance_matrix = dist
        
        return dist, metrics
    
    def get_path(self, inicio, fim):
        if self.next_matrix is None:
            return []
        
        inicio_idx = self.vertices.index(inicio)
        fim_idx = self.vertices.index(fim)
        
        if self.next_matrix[inicio_idx][fim_idx] is None:
            return []
        
        caminho = [inicio]
        atual = inicio_idx
        while atual != fim_idx:
            atual = self.next_matrix[atual][fim_idx]
            caminho.append(self.vertices[atual])
        
        return caminho

class MetricsCollector:
    def __init__(self):
        self.results = []
    
    def add_result(self, example_name, algorithm, num_vertices, num_edges, 
                   distance, path_length, metrics):
        result = {
            'timestamp': datetime.now().isoformat(),
            'example': example_name,
            'algorithm': algorithm,
            'num_vertices': num_vertices,
            'num_edges': num_edges,
            'distance': distance,
            'path_length': path_length,
            'execution_time_ms': metrics['execution_time'],
            'memory_usage_kb': metrics['memory_usage'],
            'comparisons': metrics['comparisons'],
            'iterations': metrics['iterations']
        }
        self.results.append(result)
    
    def save_to_csv(self, filename='algorithm_comparison.csv'):
        if not self.results:
            print("Nenhum resultado para salvar.")
            return
        
        fieldnames = ['timestamp', 'example', 'algorithm', 'num_vertices', 'num_edges',
                     'distance', 'path_length', 'execution_time_ms', 'memory_usage_kb', 
                     'comparisons', 'iterations']

        with open(f"{base_path}/{filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Métricas salvas em {filename}")

def exemplo_1_basico(collector):
    """Exemplo 1: Grafo simples triangular - 3 vértices"""
    print("=== EXEMPLO 1: Grafo Triangular (3 vértices) ===")
    
    vertices = ['A', 'B', 'C']
    conexoes = [('A', 'B', 10), ('B', 'C', 5), ('A', 'C', 20)]
    
    # Dijkstra
    g_dij = DijkstraGraph()
    for v in vertices:
        g_dij.add_vertex(v)
    for origem, destino, peso in conexoes:
        g_dij.add_edge(origem, destino, peso)
    
    distances, predecessors, path, metrics_dij = g_dij.dijkstra('A', 'C')
    dist_dij = distances['C']
    
    collector.add_result('Exemplo_1_Triangular', 'Dijkstra', len(vertices), 
                        len(conexoes), dist_dij, len(path), metrics_dij)
    
    # Floyd-Warshall
    g_fw = FloydWarshallGraph()
    g_fw.add_vertices(vertices)
    for origem, destino, peso in conexoes:
        g_fw.add_edge(origem, destino, peso)
    
    dist_matrix, metrics_fw = g_fw.floyd_warshall()
    dist_fw = dist_matrix[0][2]
    path_fw = g_fw.get_path('A', 'C')
    
    collector.add_result('Exemplo_1_Triangular', 'Floyd_Warshall', len(vertices), 
                        len(conexoes), dist_fw, len(path_fw), metrics_fw)
    
    salvar_grafo_como_imagem("Exemplo_1_Triangular", vertices, conexoes, f"{base_path}/graphs")
    salvar_grafo_com_caminho("Exemplo_1_Triangular", vertices, conexoes, f"{base_path}/graphs", path) 

    print(f"Dijkstra: Distância = {dist_dij}, Caminho = {path}, Tempo = {metrics_dij['execution_time']:.3f}ms")
    print(f"Floyd-Warshall: Distância = {dist_fw}, Caminho = {path_fw}, Tempo = {metrics_fw['execution_time']:.3f}ms")
    print()

def exemplo_2_medio(collector):
    """Exemplo 2: Rede urbana - 8 vértices"""
    print("=== EXEMPLO 2: Rede Urbana (8 vértices) ===")
    
    vertices = ['Centro', 'Norte', 'Sul', 'Leste', 'Oeste', 'Shopping', 'Aeroporto', 'Universidade']
    conexoes = [
        ('Centro', 'Norte', 15), ('Centro', 'Sul', 12), ('Centro', 'Leste', 18),
        ('Centro', 'Oeste', 10), ('Norte', 'Shopping', 8), ('Sul', 'Aeroporto', 25),
        ('Leste', 'Universidade', 14), ('Oeste', 'Shopping', 22), ('Shopping', 'Aeroporto', 30),
        ('Aeroporto', 'Universidade', 16), ('Norte', 'Universidade', 35)
    ]
    
    # Dijkstra
    g_dij = DijkstraGraph()
    for v in vertices:
        g_dij.add_vertex(v)
    for origem, destino, peso in conexoes:
        g_dij.add_edge(origem, destino, peso)
    
    distances, predecessors, path, metrics_dij = g_dij.dijkstra('Centro', 'Universidade')
    dist_dij = distances['Universidade']
    
    collector.add_result('Exemplo_2_Urbana', 'Dijkstra', len(vertices), 
                        len(conexoes), dist_dij, len(path), metrics_dij)
    
    # Floyd-Warshall
    g_fw = FloydWarshallGraph()
    g_fw.add_vertices(vertices)
    for origem, destino, peso in conexoes:
        g_fw.add_edge(origem, destino, peso)
    
    dist_matrix, metrics_fw = g_fw.floyd_warshall()
    dist_fw = dist_matrix[0][7]
    path_fw = g_fw.get_path('Centro', 'Universidade')
    
    collector.add_result('Exemplo_2_Urbana', 'Floyd_Warshall', len(vertices), 
                        len(conexoes), dist_fw, len(path_fw), metrics_fw)
    
    salvar_grafo_como_imagem("Exemplo_2_Urbana", vertices, conexoes, f"{base_path}/graphs")
    salvar_grafo_com_caminho("Exemplo_2_Urbana", vertices, conexoes, f"{base_path}/graphs", path)

    print(f"Dijkstra: Distância = {dist_dij}, Tempo = {metrics_dij['execution_time']:.3f}ms")
    print(f"Floyd-Warshall: Distância = {dist_fw}, Tempo = {metrics_fw['execution_time']:.3f}ms")
    print()

def exemplo_3_complexo(collector):
    """Exemplo 3: Rede logística - 15 vértices"""
    print("=== EXEMPLO 3: Rede Logística (15 vértices) ===")
    
    vertices = ['SP', 'RJ', 'BH', 'BSB', 'GO', 'MG', 'ES', 'PR', 'SC', 'RS', 'BA', 'PE', 'CE', 'AM', 'PA']
    conexoes = [
        ('SP', 'RJ', 430), ('SP', 'BH', 590), ('SP', 'BSB', 1150), ('SP', 'PR', 410),
        ('RJ', 'BH', 440), ('RJ', 'ES', 520), ('RJ', 'BA', 1200),
        ('BH', 'BSB', 740), ('BH', 'GO', 900), ('BH', 'BA', 800),
        ('BSB', 'GO', 210), ('BSB', 'BA', 1100), ('BSB', 'AM', 1950),
        ('GO', 'MG', 350), ('MG', 'ES', 300), ('PR', 'SC', 300),
        ('SC', 'RS', 460), ('BA', 'PE', 800), ('PE', 'CE', 630),
        ('CE', 'AM', 1700), ('AM', 'PA', 1300), ('PA', 'CE', 1800),
        ('ES', 'BA', 900), ('RS', 'SP', 1100), ('GO', 'BA', 1000)
    ]
    
    # Dijkstra
    g_dij = DijkstraGraph()
    for v in vertices:
        g_dij.add_vertex(v)
    for origem, destino, peso in conexoes:
        g_dij.add_edge(origem, destino, peso)
    
    distances, predecessors, path, metrics_dij = g_dij.dijkstra('SP', 'AM')
    dist_dij = distances['AM']
    
    collector.add_result('Exemplo_3_Logistica', 'Dijkstra', len(vertices), 
                        len(conexoes), dist_dij, len(path), metrics_dij)
    
    # Floyd-Warshall
    g_fw = FloydWarshallGraph()
    g_fw.add_vertices(vertices)
    for origem, destino, peso in conexoes:
        g_fw.add_edge(origem, destino, peso)
    
    dist_matrix, metrics_fw = g_fw.floyd_warshall()
    dist_fw = dist_matrix[0][13]
    path_fw = g_fw.get_path('SP', 'AM')
    
    collector.add_result('Exemplo_3_Logistica', 'Floyd_Warshall', len(vertices), 
                        len(conexoes), dist_fw, len(path_fw), metrics_fw)
    
    salvar_grafo_como_imagem("Exemplo_3_Logistica", vertices, conexoes, f"{base_path}/graphs")
    salvar_grafo_com_caminho("Exemplo_3_Logistica", vertices, conexoes, f"{base_path}/graphs", path) 

    print(f"Dijkstra: Distância = {dist_dij}, Tempo = {metrics_dij['execution_time']:.3f}ms")
    print(f"Floyd-Warshall: Distância = {dist_fw}, Tempo = {metrics_fw['execution_time']:.3f}ms")
    print()

def exemplo_4_denso(collector):
    """Exemplo 4: Grafo denso - 25 vértices com muitas conexões"""
    print("=== EXEMPLO 4: Grafo Denso (25 vértices) ===")
    
    vertices = [f'V{i}' for i in range(25)]
    
    # Criar conexões densas
    import random
    random.seed(42)
    conexoes = []
    
    for i in range(25):
        for j in range(i+1, 25):
            if random.random() < 0.4:  # 40% chance de conexão
                peso = random.randint(10, 100)
                conexoes.append((f'V{i}', f'V{j}', peso))
    
    # Dijkstra
    g_dij = DijkstraGraph()
    for v in vertices:
        g_dij.add_vertex(v)
    for origem, destino, peso in conexoes:
        g_dij.add_edge(origem, destino, peso)
    
    distances, predecessors, path, metrics_dij = g_dij.dijkstra('V0', 'V24')
    dist_dij = distances['V24']
    
    collector.add_result('Exemplo_4_Denso', 'Dijkstra', len(vertices), 
                        len(conexoes), dist_dij, len(path), metrics_dij)
    
    # Floyd-Warshall
    g_fw = FloydWarshallGraph()
    g_fw.add_vertices(vertices)
    for origem, destino, peso in conexoes:
        g_fw.add_edge(origem, destino, peso)
    
    dist_matrix, metrics_fw = g_fw.floyd_warshall()
    dist_fw = dist_matrix[0][24]
    path_fw = g_fw.get_path('V0', 'V24')
    
    collector.add_result('Exemplo_4_Denso', 'Floyd_Warshall', len(vertices), 
                        len(conexoes), dist_fw, len(path_fw), metrics_fw)
    
    salvar_grafo_como_imagem("Exemplo_4_Denso", vertices, conexoes, f"{base_path}/graphs")
    salvar_grafo_com_caminho("Exemplo_4_Denso", vertices, conexoes, f"{base_path}/graphs", path)

    print(f"Dijkstra: Distância = {dist_dij}, Tempo = {metrics_dij['execution_time']:.3f}ms")
    print(f"Floyd-Warshall: Distância = {dist_fw}, Tempo = {metrics_fw['execution_time']:.3f}ms")
    print(f"Conexões: {len(conexoes)}")
    print()

def exemplo_5_muito_complexo(collector):
    """Exemplo 5: Rede muito complexa - 50 vértices"""
    print("=== EXEMPLO 5: Rede Muito Complexa (50 vértices) ===")
    
    vertices = [f'N{i}' for i in range(50)]
    
    # Criar uma rede complexa
    import random
    random.seed(123)
    conexoes = []
    
    # Conexões em cadeia
    for i in range(49):
        peso = random.randint(5, 50)
        conexoes.append((f'N{i}', f'N{i+1}', peso))
    
    # Conexões de atalho
    for i in range(0, 50, 5):
        for j in range(i+10, min(50, i+20)):
            if random.random() < 0.3:
                peso = random.randint(20, 80)
                conexoes.append((f'N{i}', f'N{j}', peso))
    
    # Conexões aleatórias adicionais
    for _ in range(100):
        i, j = random.sample(range(50), 2)
        peso = random.randint(15, 90)
        conexoes.append((f'N{i}', f'N{j}', peso))
    
    # Dijkstra
    g_dij = DijkstraGraph()
    for v in vertices:
        g_dij.add_vertex(v)
    for origem, destino, peso in conexoes:
        g_dij.add_edge(origem, destino, peso)
    
    distances, predecessors, path, metrics_dij = g_dij.dijkstra('N0', 'N49')
    dist_dij = distances['N49']
    
    collector.add_result('Exemplo_5_Complexo', 'Dijkstra', len(vertices), 
                        len(conexoes), dist_dij, len(path), metrics_dij)
    
    # Floyd-Warshall
    g_fw = FloydWarshallGraph()
    g_fw.add_vertices(vertices)
    for origem, destino, peso in conexoes:
        g_fw.add_edge(origem, destino, peso)
    
    dist_matrix, metrics_fw = g_fw.floyd_warshall()
    dist_fw = dist_matrix[0][49]
    path_fw = g_fw.get_path('N0', 'N49')
    
    collector.add_result('Exemplo_5_Complexo', 'Floyd_Warshall', len(vertices), 
                        len(conexoes), dist_fw, len(path_fw), metrics_fw)
    
    salvar_grafo_como_imagem("Exemplo_5_Complexo", vertices, conexoes, f"{base_path}/graphs")
    salvar_grafo_com_caminho("Exemplo_5_Complexo", vertices, conexoes, f"{base_path}/graphs", path)

    print(f"Dijkstra: Distância = {dist_dij}, Tempo = {metrics_dij['execution_time']:.3f}ms")
    print(f"Floyd-Warshall: Distância = {dist_fw}, Tempo = {metrics_fw['execution_time']:.3f}ms")
    print(f"Conexões: {len(conexoes)}")
    print()

def exemplo_6_denso_floyd(collector):
    """Exemplo 6: Grafo Denso com 20 vértices - Ideal para Floyd-Warshall"""
    print("=== EXEMPLO 6: Grafo Denso (20 vértices) ===")
    
    vertices = [f'V{i}' for i in range(20)]
    conexoes = []

    import random
    random.seed(2025)

    # Gerar um grafo denso: ~80% de conexões possíveis
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            if random.random() < 0.8:  # 80% chance de conexão
                peso = random.randint(1, 20)
                conexoes.append((vertices[i], vertices[j], peso))

    # Dijkstra
    g_dij = DijkstraGraph()
    for v in vertices:
        g_dij.add_vertex(v)
    for origem, destino, peso in conexoes:
        g_dij.add_edge(origem, destino, peso)

    distances, predecessors, path, metrics_dij = g_dij.dijkstra('V0', 'V19')
    dist_dij = distances['V19']

    collector.add_result('Exemplo_6_Denso_Floyd', 'Dijkstra', len(vertices), 
                         len(conexoes), dist_dij, len(path), metrics_dij)

    # Floyd-Warshall
    g_fw = FloydWarshallGraph()
    g_fw.add_vertices(vertices)
    for origem, destino, peso in conexoes:
        g_fw.add_edge(origem, destino, peso)

    dist_matrix, metrics_fw = g_fw.floyd_warshall()
    dist_fw = dist_matrix[0][19]
    path_fw = g_fw.get_path('V0', 'V19')

    collector.add_result('Exemplo_6_Denso_Floyd', 'Floyd_Warshall', len(vertices), 
                         len(conexoes), dist_fw, len(path_fw), metrics_fw)

    salvar_grafo_como_imagem("Exemplo_6_Denso_Floyd", vertices, conexoes, f"{base_path}/graphs")
    # salvar_grafo_com_caminho("Exemplo_6_Denso_Floyd", vertices, conexoes, f"{base_path}/graphs", path)

    print(f"Dijkstra: Distância = {dist_dij}, Tempo = {metrics_dij['execution_time']:.3f}ms")
    print(f"Floyd-Warshall: Distância = {dist_fw}, Tempo = {metrics_fw['execution_time']:.3f}ms")
    print(f"Total de conexões: {len(conexoes)}")
    print()

def exemplo_7_todos_os_pares(collector):
    """Exemplo 7: Cálculo de todos os pares com Dijkstra x Floyd-Warshall"""
    print("=== EXEMPLO 7: Todos os pares (100 vértices) ===")
    
    import random
    random.seed(777)
    
    vertices = [f'N{i}' for i in range(100)]
    conexoes = []
    
    # Grafo denso (~60% de todas as conexões)
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if random.random() < 0.6:
                peso = random.randint(1, 50)
                conexoes.append((vertices[i], vertices[j], peso))
    
    # Floyd-Warshall (uma vez só)
    g_fw = FloydWarshallGraph()
    g_fw.add_vertices(vertices)
    for origem, destino, peso in conexoes:
        g_fw.add_edge(origem, destino, peso)
    
    dist_matrix, metrics_fw = g_fw.floyd_warshall()
    total_fw_distance = sum(sum(row) for row in dist_matrix if all(isinstance(v, (int, float)) for v in row))

    collector.add_result('Exemplo_7_Todos_Pares', 'Floyd_Warshall', len(vertices),
                         len(conexoes), total_fw_distance, 0, metrics_fw)

    # Dijkstra para todos os pares
    g_dij = DijkstraGraph()
    for v in vertices:
        g_dij.add_vertex(v)
    for origem, destino, peso in conexoes:
        g_dij.add_edge(origem, destino, peso)

    execution_time_total = 0
    memory_usage_total = 0
    comparisons_total = 0
    iterations_total = 0
    total_distance = 0

    for i in vertices:
        distances, _, _, metrics = g_dij.dijkstra(i)
        total_distance += sum(distances[v] for v in vertices if distances[v] < float('inf'))
        comparisons_total += metrics['comparisons']
        execution_time_total += metrics['execution_time']
        memory_usage_total += metrics['memory_usage']
        iterations_total += metrics['iterations']


    metrics_dij_all_pairs = {
        'execution_time': execution_time_total,
        'memory_usage': memory_usage_total,
        'comparisons': comparisons_total,
        'iterations': iterations_total,
    }

    collector.add_result('Exemplo_7_Todos_Pares', 'Dijkstra', len(vertices),
                         len(conexoes), total_distance, 0, metrics_dij_all_pairs)

    salvar_grafo_como_imagem("Exemplo_7_Todos_Pares", vertices, conexoes, f"{base_path}/graphs")

    print(f"Dijkstra (todos os pares): Tempo = {execution_time_total:.2f}ms, Memória = {memory_usage_total:.2f}KB")
    print(f"Floyd-Warshall: Tempo = {metrics_fw['execution_time']:.2f}ms, Memória = {metrics_fw['memory_usage']:.2f}KB")
    print()

def comparacao_completa():
    """Executa todos os exemplos e salva as métricas"""
    print("=" * 60)
    print("COMPARAÇÃO COMPLETA: DIJKSTRA vs FLOYD-WARSHALL")
    print("=" * 60)
    
    collector = MetricsCollector()
    
    exemplo_1_basico(collector)
    exemplo_2_medio(collector)
    exemplo_3_complexo(collector)
    exemplo_4_denso(collector)
    exemplo_5_muito_complexo(collector)
    # exemplo_6_denso_floyd(collector)
    exemplo_7_todos_os_pares(collector)
    
    # Salvar métricas em CSV
    collector.save_to_csv('algorithm_comparison.csv')
    
    print("=" * 60)
    print("RESUMO DAS MÉTRICAS:")
    print("• Arquivo CSV salvo: algorithm_comparison.csv")
    print("• Métricas coletadas: tempo de execução, uso de memória, comparações, iterações")
    print("• Dados salvos com timestamp para análise posterior")
    print("=" * 60)
    
    # Mostrar resumo dos resultados
    print("\nRESUMO COMPARATIVO:")
    dijkstra_times = [r['execution_time_ms'] for r in collector.results if r['algorithm'] == 'Dijkstra']
    floyd_times = [r['execution_time_ms'] for r in collector.results if r['algorithm'] == 'Floyd_Warshall']
    
    print(f"Dijkstra - Tempo médio: {sum(dijkstra_times)/len(dijkstra_times):.3f}ms")
    print(f"Floyd-Warshall - Tempo médio: {sum(floyd_times)/len(floyd_times):.3f}ms")
    
    dijkstra_memory = [r['memory_usage_kb'] for r in collector.results if r['algorithm'] == 'Dijkstra']
    floyd_memory = [r['memory_usage_kb'] for r in collector.results if r['algorithm'] == 'Floyd_Warshall']
    
    print(f"Dijkstra - Memória média: {sum(dijkstra_memory)/len(dijkstra_memory):.2f}KB")
    print(f"Floyd-Warshall - Memória média: {sum(floyd_memory)/len(floyd_memory):.2f}KB")

def analyze_csv_results(filename='algorithm_comparison.csv'):
    """
    Lê o arquivo CSV e gera gráficos comparativos entre Dijkstra e Floyd-Warshall
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    try:
        # Ler dados do CSV
        df = pd.read_csv(f"{base_path}/{filename}")
        print(f"Dados carregados: {len(df)} registros")
        
        # Configurar estilo dos gráficos
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Criar subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comparação Dijkstra vs Floyd-Warshall', fontsize=16, fontweight='bold')
        
        # 1. Tempo de Execução por Exemplo
        ax1 = axes[0, 0]
        df_pivot = df.pivot(index='example', columns='algorithm', values='execution_time_ms')
        df_pivot.plot(kind='bar', ax=ax1, rot=45)
        ax1.set_title('Tempo de Execução (ms)')
        ax1.set_ylabel('Tempo (ms)')
        ax1.legend(title='Algoritmo')
        ax1.grid(True, alpha=0.3)
        
        # 2. Uso de Memória por Exemplo
        ax2 = axes[0, 1]
        df_pivot_mem = df.pivot(index='example', columns='algorithm', values='memory_usage_kb')
        df_pivot_mem.plot(kind='bar', ax=ax2, rot=45)
        ax2.set_title('Uso de Memória (KB)')
        ax2.set_ylabel('Memória (KB)')
        ax2.legend(title='Algoritmo')
        ax2.grid(True, alpha=0.3)
        
        # 3. Número de Comparações
        ax3 = axes[0, 2]
        df_pivot_comp = df.pivot(index='example', columns='algorithm', values='comparisons')
        df_pivot_comp.plot(kind='bar', ax=ax3, rot=45)
        ax3.set_title('Número de Comparações')
        ax3.set_ylabel('Comparações')
        ax3.legend(title='Algoritmo')
        ax3.grid(True, alpha=0.3)
        
        # 4. Número de Iterações
        ax4 = axes[1, 0]
        df_pivot_iter = df.pivot(index='example', columns='algorithm', values='iterations')
        df_pivot_iter.plot(kind='bar', ax=ax4, rot=45)
        ax4.set_title('Número de Iterações')
        ax4.set_ylabel('Iterações')
        ax4.legend(title='Algoritmo')
        ax4.grid(True, alpha=0.3)
        
        # 5. Escalabilidade - Tempo vs Número de Vértices
        ax5 = axes[1, 1]
        for algorithm in df['algorithm'].unique():
            subset = df[df['algorithm'] == algorithm]
            ax5.scatter(subset['num_vertices'], subset['execution_time_ms'], 
                       label=algorithm, alpha=0.7, s=60)
        ax5.set_title('Escalabilidade: Tempo vs Vértices')
        ax5.set_xlabel('Número de Vértices')
        ax5.set_ylabel('Tempo (ms)')
        ax5.legend(title='Algoritmo')
        ax5.grid(True, alpha=0.3)
        
        # 6. Eficiência - Memória vs Vértices
        ax6 = axes[1, 2]
        for algorithm in df['algorithm'].unique():
            subset = df[df['algorithm'] == algorithm]
            ax6.scatter(subset['num_vertices'], subset['memory_usage_kb'], 
                       label=algorithm, alpha=0.7, s=60)
        ax6.set_title('Eficiência: Memória vs Vértices')
        ax6.set_xlabel('Número de Vértices')
        ax6.set_ylabel('Memória (KB)')
        ax6.legend(title='Algoritmo')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{base_path}/algorithm_comparison_charts.png", dpi=300, bbox_inches='tight')
        
        # Gráfico adicional - Comparação direta em linha
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Tempo
        plt.subplot(2, 2, 1)
        examples = df['example'].unique()
        dijkstra_times = []
        floyd_times = []
        
        for example in examples:
            dij_time = df[(df['example'] == example) & (df['algorithm'] == 'Dijkstra')]['execution_time_ms'].iloc[0]
            floyd_time = df[(df['example'] == example) & (df['algorithm'] == 'Floyd_Warshall')]['execution_time_ms'].iloc[0]
            dijkstra_times.append(dij_time)
            floyd_times.append(floyd_time)
        
        plt.plot(range(len(examples)), dijkstra_times, 'o-', label='Dijkstra', linewidth=2)
        plt.plot(range(len(examples)), floyd_times, 's-', label='Floyd-Warshall', linewidth=2)
        plt.title('Evolução do Tempo de Execução')
        plt.xlabel('Exemplos')
        plt.ylabel('Tempo (ms)')
        plt.xticks(range(len(examples)), [e.replace('Exemplo_', '') for e in examples], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Memória
        plt.subplot(2, 2, 2)
        dijkstra_memory = []
        floyd_memory = []
        
        for example in examples:
            dij_mem = df[(df['example'] == example) & (df['algorithm'] == 'Dijkstra')]['memory_usage_kb'].iloc[0]
            floyd_mem = df[(df['example'] == example) & (df['algorithm'] == 'Floyd_Warshall')]['memory_usage_kb'].iloc[0]
            dijkstra_memory.append(dij_mem)
            floyd_memory.append(floyd_mem)
        
        plt.plot(range(len(examples)), dijkstra_memory, 'o-', label='Dijkstra', linewidth=2)
        plt.plot(range(len(examples)), floyd_memory, 's-', label='Floyd-Warshall', linewidth=2)
        plt.title('Evolução do Uso de Memória')
        plt.xlabel('Exemplos')
        plt.ylabel('Memória (KB)')
        plt.xticks(range(len(examples)), [e.replace('Exemplo_', '') for e in examples], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Comparações
        plt.subplot(2, 2, 3)
        dijkstra_comp = []
        floyd_comp = []
        
        for example in examples:
            dij_comp = df[(df['example'] == example) & (df['algorithm'] == 'Dijkstra')]['comparisons'].iloc[0]
            floyd_comp = df[(df['example'] == example) & (df['algorithm'] == 'Floyd_Warshall')]['comparisons'].iloc[0]
            dijkstra_comp.append(dij_comp)
            floyd_comp.append(floyd_comp)
        
        plt.plot(range(len(examples)), dijkstra_comp, 'o-', label='Dijkstra', linewidth=2)
        plt.plot(range(len(examples)), floyd_comp, 's-', label='Floyd-Warshall', linewidth=2)
        plt.title('Número de Comparações')
        plt.xlabel('Exemplos')
        plt.ylabel('Comparações')
        plt.xticks(range(len(examples)), [e.replace('Exemplo_', '') for e in examples], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Iterações
        plt.subplot(2, 2, 4)
        dijkstra_iter = []
        floyd_iter = []
        
        for example in examples:
            dij_iter = df[(df['example'] == example) & (df['algorithm'] == 'Dijkstra')]['iterations'].iloc[0]
            floyd_iter = df[(df['example'] == example) & (df['algorithm'] == 'Floyd_Warshall')]['iterations'].iloc[0]
            dijkstra_iter.append(dij_iter)
            floyd_iter.append(floyd_iter)
        
        plt.plot(range(len(examples)), dijkstra_iter, 'o-', label='Dijkstra', linewidth=2)
        plt.plot(range(len(examples)), floyd_iter, 's-', label='Floyd-Warshall', linewidth=2)
        plt.title('Número de Iterações')
        plt.xlabel('Exemplos')
        plt.ylabel('Iterações')
        plt.xticks(range(len(examples)), [e.replace('Exemplo_', '') for e in examples], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{base_path}/algorithm_trends.png", dpi=300, bbox_inches='tight')
        
        # Imprimir estatísticas resumidas
        print("\n" + "="*60)
        print("ESTATÍSTICAS RESUMIDAS")
        print("="*60)
        
        for metric in ['execution_time_ms', 'memory_usage_kb', 'comparisons', 'iterations']:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            for algorithm in df['algorithm'].unique():
                subset = df[df['algorithm'] == algorithm][metric]
                print(f"  {algorithm}: Média = {subset.mean():.2f}, Desvio = {subset.std():.2f}")
        
        print(f"\nGráficos salvos:")
        print("• algorithm_comparison_charts.png")
        print("• algorithm_trends.png")
        
    except FileNotFoundError:
        print(f"Arquivo {filename} não encontrado. Execute a comparação primeiro.")
    except Exception as e:
        print(f"Erro ao processar dados: {e}")

def setup_plot_style():
    """Configurar estilo dos gráficos"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def load_data(filename='algorithm_comparison.csv'):
    """Carregar dados do CSV"""
    try:
        df = pd.read_csv(f"{base_path}/{filename}")
        print(f"Dados carregados: {len(df)} registros")
        return df
    except FileNotFoundError:
        print(f"Arquivo {filename} não encontrado.")
        return None

def create_execution_time_chart(df, output_dir):
    """Gráfico 1: Tempo de Execução por Exemplo"""
    plt.figure(figsize=(12, 7))
    
    # Preparar dados
    examples = df['example'].unique()
    dijkstra_times = []
    floyd_times = []
    
    for example in examples:
        dij_time = df[(df['example'] == example) & (df['algorithm'] == 'Dijkstra')]['execution_time_ms'].iloc[0]
        floyd_time = df[(df['example'] == example) & (df['algorithm'] == 'Floyd_Warshall')]['execution_time_ms'].iloc[0]
        dijkstra_times.append(dij_time)
        floyd_times.append(floyd_time)
    
    # Criar gráfico de barras
    x = np.arange(len(examples))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, dijkstra_times, width, label='Dijkstra', color='#2E86C1', alpha=0.8)
    bars2 = plt.bar(x + width/2, floyd_times, width, label='Floyd-Warshall', color='#E74C3C', alpha=0.8)
    
    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}ms', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}ms', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Exemplos de Grafos')
    plt.ylabel('Tempo de Execução (ms)')
    plt.title('Comparação de Tempo de Execução: Dijkstra vs Floyd-Warshall')
    plt.xticks(x, [e.replace('Exemplo_', '').replace('_', ' ') for e in examples], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_tempo_execucao.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_memory_usage_chart(df, output_dir):
    """Gráfico 2: Uso de Memória por Exemplo"""
    plt.figure(figsize=(12, 7))
    
    examples = df['example'].unique()
    dijkstra_memory = []
    floyd_memory = []
    
    for example in examples:
        dij_mem = df[(df['example'] == example) & (df['algorithm'] == 'Dijkstra')]['memory_usage_kb'].iloc[0]
        floyd_mem = df[(df['example'] == example) & (df['algorithm'] == 'Floyd_Warshall')]['memory_usage_kb'].iloc[0]
        dijkstra_memory.append(dij_mem)
        floyd_memory.append(floyd_mem)
    
    x = np.arange(len(examples))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, dijkstra_memory, width, label='Dijkstra', color='#28B463', alpha=0.8)
    bars2 = plt.bar(x + width/2, floyd_memory, width, label='Floyd-Warshall', color='#F39C12', alpha=0.8)
    
    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}KB', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}KB', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Exemplos de Grafos')
    plt.ylabel('Uso de Memória (KB)')
    plt.title('Comparação de Uso de Memória: Dijkstra vs Floyd-Warshall')
    plt.xticks(x, [e.replace('Exemplo_', '').replace('_', ' ') for e in examples], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_uso_memoria.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_comparisons_chart(df, output_dir):
    """Gráfico 3: Número de Comparações"""
    plt.figure(figsize=(12, 7))
    
    examples = df['example'].unique()
    dijkstra_comp = []
    floyd_comp_array = []
    
    for example in examples:
        dij_comp = df[(df['example'] == example) & (df['algorithm'] == 'Dijkstra')]['comparisons'].iloc[0]
        floyd_comp = df[(df['example'] == example) & (df['algorithm'] == 'Floyd_Warshall')]['comparisons'].iloc[0]
        dijkstra_comp.append(dij_comp)
        floyd_comp_array.append(floyd_comp)
    
    x = np.arange(len(examples))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, dijkstra_comp, width, label='Dijkstra', color='#8E44AD', alpha=0.8)
    bars2 = plt.bar(x + width/2, floyd_comp_array, width, label='Floyd-Warshall', color='#D35400', alpha=0.8)
    
    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(dijkstra_comp + floyd_comp_array)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(dijkstra_comp + floyd_comp_array)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Exemplos de Grafos')
    plt.ylabel('Número de Comparações')
    plt.title('Comparação do Número de Comparações: Dijkstra vs Floyd-Warshall')
    plt.xticks(x, [e.replace('Exemplo_', '').replace('_', ' ') for e in examples], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_comparacoes.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_iterations_chart(df, output_dir):
    """Gráfico 4: Número de Iterações"""
    plt.figure(figsize=(12, 7))
    
    examples = df['example'].unique()
    dijkstra_iter = []
    floyd_iter_array = []
    
    for example in examples:
        dij_iter = df[(df['example'] == example) & (df['algorithm'] == 'Dijkstra')]['iterations'].iloc[0]
        floyd_iter = df[(df['example'] == example) & (df['algorithm'] == 'Floyd_Warshall')]['iterations'].iloc[0]
        dijkstra_iter.append(dij_iter)
        floyd_iter_array.append(floyd_iter)
    
    x = np.arange(len(examples))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, dijkstra_iter, width, label='Dijkstra', color='#16A085', alpha=0.8)
    bars2 = plt.bar(x + width/2, floyd_iter_array, width, label='Floyd-Warshall', color='#C0392B', alpha=0.8)
    
    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(dijkstra_iter + floyd_iter_array)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(dijkstra_iter + floyd_iter_array)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Exemplos de Grafos')
    plt.ylabel('Número de Iterações')
    plt.title('Comparação do Número de Iterações: Dijkstra vs Floyd-Warshall')
    plt.xticks(x, [e.replace('Exemplo_', '').replace('_', ' ') for e in examples], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_iteracoes.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_scalability_time_chart(df, output_dir):
    """Gráfico 5: Escalabilidade - Tempo vs Número de Vértices (Linha)"""
    plt.figure(figsize=(12, 8))
    
    # Ordenar por número de vértices
    df_sorted = df.sort_values('num_vertices')
    
    dijkstra_data = df_sorted[df_sorted['algorithm'] == 'Dijkstra']
    floyd_data = df_sorted[df_sorted['algorithm'] == 'Floyd_Warshall']
    
    plt.plot(dijkstra_data['num_vertices'], dijkstra_data['execution_time_ms'], 
             'o-', label='Dijkstra', linewidth=3, markersize=8, color='#2E86C1')
    plt.plot(floyd_data['num_vertices'], floyd_data['execution_time_ms'], 
             's-', label='Floyd-Warshall', linewidth=3, markersize=8, color='#E74C3C')
    
    # Adicionar curvas teóricas de complexidade
    vertices = np.array(sorted(df['num_vertices'].unique()))
    
    # Dijkstra: O((V + E) log V) - aproximamos como O(V²log V) para grafo denso
    dijkstra_theoretical = vertices**2 * np.log2(vertices) * 0.001  # Fator de escala
    
    # Floyd-Warshall: O(V³)
    floyd_theoretical = vertices**3 * 0.00001  # Fator de escala
    
    plt.plot(vertices, dijkstra_theoretical, '--', alpha=0.7, 
             label='Dijkstra Teórico O(V²log V)', color='#5DADE2')
    plt.plot(vertices, floyd_theoretical, '--', alpha=0.7, 
             label='Floyd-Warshall Teórico O(V³)', color='#F1948A')
    
    plt.xlabel('Número de Vértices (V)')
    plt.ylabel('Tempo de Execução (ms)')
    plt.title('Análise de Complexidade: Tempo de Execução vs Número de Vértices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar anotações
    for i, (v, t) in enumerate(zip(dijkstra_data['num_vertices'], dijkstra_data['execution_time_ms'])):
        plt.annotate(f'{t:.2f}ms', (v, t), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    for i, (v, t) in enumerate(zip(floyd_data['num_vertices'], floyd_data['execution_time_ms'])):
        plt.annotate(f'{t:.2f}ms', (v, t), xytext=(5, -15), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_escalabilidade_tempo.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_scalability_memory_chart(df, output_dir):
    """Gráfico 6: Escalabilidade - Memória vs Número de Vértices (Linha)"""
    plt.figure(figsize=(12, 8))
    
    df_sorted = df.sort_values('num_vertices')
    
    dijkstra_data = df_sorted[df_sorted['algorithm'] == 'Dijkstra']
    floyd_data = df_sorted[df_sorted['algorithm'] == 'Floyd_Warshall']
    
    plt.plot(dijkstra_data['num_vertices'], dijkstra_data['memory_usage_kb'], 
             'o-', label='Dijkstra', linewidth=3, markersize=8, color='#28B463')
    plt.plot(floyd_data['num_vertices'], floyd_data['memory_usage_kb'], 
             's-', label='Floyd-Warshall', linewidth=3, markersize=8, color='#F39C12')
    
    # Adicionar curvas teóricas de complexidade de memória
    vertices = np.array(sorted(df['num_vertices'].unique()))
    
    # Dijkstra: O(V) para distâncias + O(E) para grafo
    dijkstra_memory_theoretical = vertices * 0.5  # Fator de escala
    
    # Floyd-Warshall: O(V²) para matriz de distâncias
    floyd_memory_theoretical = vertices**2 * 0.01  # Fator de escala
    
    plt.plot(vertices, dijkstra_memory_theoretical, '--', alpha=0.7, 
             label='Dijkstra Teórico O(V)', color='#58D68D')
    plt.plot(vertices, floyd_memory_theoretical, '--', alpha=0.7, 
             label='Floyd-Warshall Teórico O(V²)', color='#F8C471')
    
    plt.xlabel('Número de Vértices (V)')
    plt.ylabel('Uso de Memória (KB)')
    plt.title('Análise de Complexidade: Uso de Memória vs Número de Vértices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar anotações
    for i, (v, m) in enumerate(zip(dijkstra_data['num_vertices'], dijkstra_data['memory_usage_kb'])):
        plt.annotate(f'{m:.1f}KB', (v, m), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    for i, (v, m) in enumerate(zip(floyd_data['num_vertices'], floyd_data['memory_usage_kb'])):
        plt.annotate(f'{m:.1f}KB', (v, m), xytext=(5, -15), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_escalabilidade_memoria.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_complexity_analysis_chart(df, output_dir):
    """Gráfico 7: Análise de Complexidade Comparativa"""
    plt.figure(figsize=(14, 10))
    
    # Criar subplot com 2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    df_sorted = df.sort_values('num_vertices')
    dijkstra_data = df_sorted[df_sorted['algorithm'] == 'Dijkstra']
    floyd_data = df_sorted[df_sorted['algorithm'] == 'Floyd_Warshall']
    
    # Subplot 1: Tempo de execução
    ax1.plot(dijkstra_data['num_vertices'], dijkstra_data['execution_time_ms'], 
             'o-', label='Dijkstra', linewidth=2, markersize=6, color='#2E86C1')
    ax1.plot(floyd_data['num_vertices'], floyd_data['execution_time_ms'], 
             's-', label='Floyd-Warshall', linewidth=2, markersize=6, color='#E74C3C')
    ax1.set_xlabel('Número de Vértices')
    ax1.set_ylabel('Tempo (ms)')
    ax1.set_title('Tempo de Execução')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Uso de memória
    ax2.plot(dijkstra_data['num_vertices'], dijkstra_data['memory_usage_kb'], 
             'o-', label='Dijkstra', linewidth=2, markersize=6, color='#28B463')
    ax2.plot(floyd_data['num_vertices'], floyd_data['memory_usage_kb'], 
             's-', label='Floyd-Warshall', linewidth=2, markersize=6, color='#F39C12')
    ax2.set_xlabel('Número de Vértices')
    ax2.set_ylabel('Memória (KB)')
    ax2.set_title('Uso de Memória')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Comparações
    ax3.plot(dijkstra_data['num_vertices'], dijkstra_data['comparisons'], 
             'o-', label='Dijkstra', linewidth=2, markersize=6, color='#8E44AD')
    ax3.plot(floyd_data['num_vertices'], floyd_data['comparisons'], 
             's-', label='Floyd-Warshall', linewidth=2, markersize=6, color='#D35400')
    ax3.set_xlabel('Número de Vértices')
    ax3.set_ylabel('Comparações')
    ax3.set_title('Número de Comparações')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Iterações
    ax4.plot(dijkstra_data['num_vertices'], dijkstra_data['iterations'], 
             'o-', label='Dijkstra', linewidth=2, markersize=6, color='#16A085')
    ax4.plot(floyd_data['num_vertices'], floyd_data['iterations'], 
             's-', label='Floyd-Warshall', linewidth=2, markersize=6, color='#C0392B')
    ax4.set_xlabel('Número de Vértices')
    ax4.set_ylabel('Iterações')
    ax4.set_title('Número de Iterações')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Análise Completa de Complexidade: Dijkstra vs Floyd-Warshall', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_analise_complexidade.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_ratio_chart(df, output_dir):
    """Gráfico 8: Razão de Performance (Floyd/Dijkstra)"""
    plt.figure(figsize=(12, 8))
    
    examples = df['example'].unique()
    time_ratios = []
    memory_ratios = []
    vertices = []
    
    for example in examples:
        dij_time = df[(df['example'] == example) & (df['algorithm'] == 'Dijkstra')]['execution_time_ms'].iloc[0]
        floyd_time = df[(df['example'] == example) & (df['algorithm'] == 'Floyd_Warshall')]['execution_time_ms'].iloc[0]
        dij_memory = df[(df['example'] == example) & (df['algorithm'] == 'Dijkstra')]['memory_usage_kb'].iloc[0]
        floyd_memory = df[(df['example'] == example) & (df['algorithm'] == 'Floyd_Warshall')]['memory_usage_kb'].iloc[0]
        num_vertices = df[df['example'] == example]['num_vertices'].iloc[0]
        
        time_ratios.append(floyd_time / dij_time if dij_time > 0 else 0)
        memory_ratios.append(floyd_memory / dij_memory if dij_memory > 0 else 0)
        vertices.append(num_vertices)
    
    # Criar dois eixos Y
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color = 'tab:red'
    ax1.set_xlabel('Número de Vértices')
    ax1.set_ylabel('Razão de Tempo (Floyd/Dijkstra)', color=color)
    line1 = ax1.plot(vertices, time_ratios, 'o-', color=color, linewidth=3, markersize=8, label='Razão de Tempo')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Razão de Memória (Floyd/Dijkstra)', color=color)
    line2 = ax2.plot(vertices, memory_ratios, 's-', color=color, linewidth=3, markersize=8, label='Razão de Memória')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Adicionar linha de referência (y=1)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Paridade (1:1)')
    
    plt.title('Razão de Performance: Floyd-Warshall / Dijkstra\n(Valores > 1 indicam que Floyd-Warshall é pior)')
    
    # Adicionar anotações
    for i, (v, tr, mr) in enumerate(zip(vertices, time_ratios, memory_ratios)):
        ax1.annotate(f'{tr:.1f}x', (v, tr), xytext=(5, 5), textcoords='offset points', fontsize=9, color='red')
        ax2.annotate(f'{mr:.1f}x', (v, mr), xytext=(5, -15), textcoords='offset points', fontsize=9, color='blue')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_razao_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_theoretical_complexity_chart(output_dir):
    """Gráfico 9: Comparação Teórica de Complexidade"""
    plt.figure(figsize=(12, 8))
    
    # Gerar dados teóricos
    vertices = np.arange(5, 51, 5)
    
    # Dijkstra: O((V + E) log V) - para grafo denso E ≈ V², então O(V² log V)
    dijkstra_complexity = vertices**2 * np.log2(vertices)
    
    # Floyd-Warshall: O(V³)
    floyd_complexity = vertices**3
    
    # Normalizar para visualização
    dijkstra_normalized = dijkstra_complexity / dijkstra_complexity[0]
    floyd_normalized = floyd_complexity / floyd_complexity[0]
    
    plt.plot(vertices, dijkstra_normalized, 'o-', label='Dijkstra O(V²log V)', 
             linewidth=3, markersize=8, color='#2E86C1')
    plt.plot(vertices, floyd_normalized, 's-', label='Floyd-Warshall O(V³)', 
             linewidth=3, markersize=8, color='#E74C3C')
    
    plt.xlabel('Número de Vértices (V)')
    plt.ylabel('Complexidade Relativa (Normalizada)')
    plt.title('Comparação Teórica de Complexidade de Tempo\n(Normalizada pelo menor valor)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Adicionar anotações em pontos específicos
    for i in [0, 4, 9]:  # Pontos específicos
        plt.annotate(f'V={vertices[i]}\nDij: {dijkstra_normalized[i]:.1f}\nFW: {floyd_normalized[i]:.1f}', 
                    xy=(vertices[i], max(dijkstra_normalized[i], floyd_normalized[i])), 
                    xytext=(10, 10), textcoords='offset points', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/09_complexidade_teorica.png", dpi=300, bbox_inches='tight')
    plt.close()

def salvar_grafo_como_imagem(nome_exemplo, vertices, conexoes, output_dir='output/graphs'):
    """
    Gera e salva uma imagem do grafo com pesos nas arestas.
    
    :param nome_exemplo: Nome do exemplo (usado como nome do arquivo)
    :param vertices: Lista de vértices
    :param conexoes: Lista de tuplas (origem, destino, peso)
    :param output_dir: Caminho onde os arquivos PNG serão salvos
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for origem, destino, peso in conexoes:
        G.add_edge(origem, destino, weight=peso)

    pos = nx.spring_layout(G, seed=42)  # Layout bonito
    weights = nx.get_edge_attributes(G, 'weight')
    
    plt.figure(figsize=(20, 16))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=800, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_color='black')

    plt.title(f"Grafo: {nome_exemplo.replace('_', ' ')}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{nome_exemplo}.png", dpi=300)
    plt.close()
    print(f"✓ Grafo salvo como imagem: {output_dir}/{nome_exemplo}.png")

def salvar_grafo_com_caminho(nome_exemplo, vertices, conexoes, output_dir='output/graphs', caminho=None):
    """
    Gera e salva imagem do grafo com destaque para o caminho ótimo, se fornecido.
    
    :param nome_exemplo: Nome do exemplo
    :param vertices: Lista de vértices
    :param conexoes: Lista de arestas (origem, destino, peso)
    :param caminho: Lista de nós no caminho ótimo (ex: ['A', 'B', 'C'])
    :param output_dir: Pasta para salvar as imagens
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for origem, destino, peso in conexoes:
        G.add_edge(origem, destino, weight=peso)
    
    pos = nx.spring_layout(G, seed=42)
    weights = nx.get_edge_attributes(G, 'weight')
    
    plt.figure(figsize=(10, 8))
    
    # Arestas do caminho
    caminho_edges = []
    if caminho and len(caminho) >= 2:
        caminho_edges = [(caminho[i], caminho[i+1]) for i in range(len(caminho)-1)]
    
    # Desenhar todas as arestas em cinza
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', width=1)

    # Destacar as arestas do caminho em vermelho
    if caminho_edges:
        nx.draw_networkx_edges(G, pos, edgelist=caminho_edges, edge_color='red', width=3)

    # Destacar os nós do caminho
    node_colors = ['orange' if n in caminho else 'lightblue' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)

    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_color='black')
    
    plt.title(f"Grafo com caminho: {nome_exemplo.replace('_', ' ')}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{nome_exemplo}_caminho.png", dpi=300)
    plt.close()
    print(f"✓ Caminho salvo: {output_dir}/{nome_exemplo}_caminho.png")

def generate_all_charts(csv_file='algorithm_comparison.csv', output_dir='charts'):
    """Gerar todos os gráficos individuais"""
    
    # Criar diretório de saída
    Path(output_dir).mkdir(exist_ok=True)
    
    # Configurar estilo
    setup_plot_style()
    
    # Carregar dados
    df = load_data(csv_file)
    if df is None:
        return
    
    print("Gerando gráficos individuais...")
    
    # Gerar cada gráfico
    create_execution_time_chart(df, output_dir)
    print("✓ Gráfico 1: Tempo de Execução")
    
    create_memory_usage_chart(df, output_dir)
    print("✓ Gráfico 2: Uso de Memória")
    
    create_comparisons_chart(df, output_dir)
    print("✓ Gráfico 3: Comparações")
    
    create_iterations_chart(df, output_dir)
    print("✓ Gráfico 4: Iterações")
    
    create_scalability_time_chart(df, output_dir)
    print("✓ Gráfico 5: Escalabilidade de Tempo")
    
    create_scalability_memory_chart(df, output_dir)
    print("✓ Gráfico 6: Escalabilidade de Memória")
    
    create_complexity_analysis_chart(df, output_dir)
    print("✓ Gráfico 7: Análise de Complexidade")
    
    create_performance_ratio_chart(df, output_dir)
    print("✓ Gráfico 8: Razão de Performance")
    
    create_theoretical_complexity_chart(output_dir)
    print("✓ Gráfico 9: Complexidade Teórica")
    
    print(f"\n🎉 Todos os gráficos foram gerados em: {output_dir}/")
    print("\nArquivos gerados:")
    for i in range(1, 10):
        filename = f"{i:02d}_*.png"
        print(f"  • {filename}")

if __name__ == "__main__":
    os.makedirs(base_path, exist_ok=True)

    comparacao_completa()
    # analyze_csv_results()
    generate_all_charts(output_dir=f"{base_path}/charts")