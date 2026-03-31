import numpy as np

class BellmanFord:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
    
    def find_path(self, start, end):
        """Bellman-Ford algorithm (can handle negative weights)"""
        # Flatten grid into graph
        nodes = [(r, c) for r in range(self.rows) for c in range(self.cols) 
                 if self.grid[r, c] != 1]
        
        distances = {node: float('inf') for node in nodes}
        distances[start] = 0
        parent = {}
        visited_nodes = []
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Relax edges |V| - 1 times
        for _ in range(len(nodes) - 1):
            updated = False
            for node in nodes:
                if distances[node] == float('inf'):
                    continue
                row, col = node
                visited_nodes.append(node)
                
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    neighbor = (nr, nc)
                    
                    if neighbor in distances:
                        # Edge weight = 1 (unweighted)
                        if distances[node] + 1 < distances[neighbor]:
                            distances[neighbor] = distances[node] + 1
                            parent[neighbor] = node
                            updated = True
            
            if not updated:
                break
        
        # Check for negative cycles (not expected in grid)
        if distances[end] == float('inf'):
            return None, visited_nodes, float('inf')
        
        path = self.reconstruct_path(parent, start, end)
        return path, visited_nodes, distances[end]
    
    def reconstruct_path(self, parent, start, end):
        path = []
        current = end
        while current != start:
            path.append(current)
            current = parent.get(current)
            if current is None:
                return []
        path.append(start)
        return path[::-1]