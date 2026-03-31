import heapq
import numpy as np

class Dijkstra:
    def __init__(self, grid):
        """
        Initialize Dijkstra's algorithm
        grid: 2D numpy array where 0 = empty, 1 = wall
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        
    def find_path(self, start, end):
        """
        Find shortest path using Dijkstra's algorithm
        Returns: (path, visited_nodes, distance)
        """
        # Priority queue: (distance, row, col)
        pq = [(0, start[0], start[1])]
        distances = np.full((self.rows, self.cols), np.inf)
        distances[start[0], start[1]] = 0
        parent = {}
        visited_nodes = []
        
        # Direction vectors: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while pq:
            dist, row, col = heapq.heappop(pq)
            current = (row, col)
            visited_nodes.append(current)
            
            # Reached destination
            if current == end:
                path = self.reconstruct_path(parent, start, end)
                return path, visited_nodes, distances[end[0], end[1]]
            
            # Explore neighbors
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                
                # Check bounds
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    # Check if not a wall
                    if self.grid[nr, nc] == 1:  # Wall
                        continue
                    
                    new_dist = dist + 1  # Unweighted grid
                    if new_dist < distances[nr, nc]:
                        distances[nr, nc] = new_dist
                        parent[(nr, nc)] = current
                        heapq.heappush(pq, (new_dist, nr, nc))
        
        return None, visited_nodes, float('inf')  # No path found
    
    def reconstruct_path(self, parent, start, end):
        """Reconstruct the path from parent dictionary"""
        path = []
        current = end
        while current != start:
            path.append(current)
            current = parent.get(current)
            if current is None:
                return []
        path.append(start)
        return path[::-1] 