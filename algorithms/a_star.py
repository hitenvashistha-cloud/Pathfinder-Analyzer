import heapq
import numpy as np

class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
    
    def heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_path(self, start, end):
        """
        Find path using A* algorithm
        Returns: (path, visited_nodes, distance)
        """
        pq = [(0, start[0], start[1])]  # (f_score, row, col)
        g_score = np.full((self.rows, self.cols), np.inf)
        g_score[start[0], start[1]] = 0
        f_score = np.full((self.rows, self.cols), np.inf)
        f_score[start[0], start[1]] = self.heuristic(start, end)
        
        parent = {}
        visited_nodes = []
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while pq:
            _, row, col = heapq.heappop(pq)
            current = (row, col)
            visited_nodes.append(current)
            
            if current == end:
                path = self.reconstruct_path(parent, start, end)
                return path, visited_nodes, g_score[end[0], end[1]]
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr, nc] == 1:
                        continue
                    
                    tentative_g = g_score[row, col] + 1
                    
                    if tentative_g < g_score[nr, nc]:
                        parent[(nr, nc)] = current
                        g_score[nr, nc] = tentative_g
                        f_score[nr, nc] = tentative_g + self.heuristic((nr, nc), end)
                        heapq.heappush(pq, (f_score[nr, nc], nr, nc))
        
        return None, visited_nodes, float('inf')
    
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