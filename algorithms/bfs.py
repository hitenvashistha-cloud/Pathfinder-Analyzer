from collections import deque
import numpy as np

class BFS:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
    
    def find_path(self, start, end):
        """BFS for unweighted shortest path"""
        queue = deque([start])
        visited = set([start])
        parent = {start: None}
        visited_nodes = []
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            current = queue.popleft()
            visited_nodes.append(current)
            
            if current == end:
                path = self.reconstruct_path(parent, start, end)
                return path, visited_nodes, len(path) - 1
            
            row, col = current
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                neighbor = (nr, nc)
                
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr, nc] != 1 and neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append(neighbor)
        
        return None, visited_nodes, float('inf')
    
    def reconstruct_path(self, parent, start, end):
        path = []
        current = end
        while current != start:
            path.append(current)
            current = parent[current]
        path.append(start)
        return path[::-1]