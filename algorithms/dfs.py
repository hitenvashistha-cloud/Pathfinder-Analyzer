"""
Depth-First Search (DFS) Algorithm
"""

class DFS:
    def __init__(self, grid):
        """
        Initialize DFS algorithm
        grid: 2D numpy array where 0 = empty, 1 = wall
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.visited = None
        self.parent = None
        self.path_found = False
        self.visited_nodes = []
        
    def find_path(self, start, end):
        """
        Find path using Depth-First Search
        Returns: (path, visited_nodes, path_length)
        
        Note: DFS does NOT guarantee shortest path
        """
        # Reset for new search
        self.visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.parent = {}
        self.visited_nodes = []
        self.path_found = False
        
        # Start DFS from start node
        self._dfs(start[0], start[1], end)
        
        # Reconstruct path if found
        if self.path_found:
            path = self._reconstruct_path(start, end)
            path_length = len(path) - 1
            return path, self.visited_nodes, path_length
        else:
            return None, self.visited_nodes, float('inf')
    
    def _dfs(self, row, col, end):
        """
        Recursive DFS implementation
        """
        # Check bounds
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        
        # Check if wall or already visited
        if self.grid[row, col] == 1 or self.visited[row][col]:
            return False
        
        # Mark current node as visited
        self.visited[row][col] = True
        current = (row, col)
        self.visited_nodes.append(current)
        
        # Check if reached destination
        if current == end:
            self.path_found = True
            return True
        
        # Define exploration order (up, right, down, left)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if not self.visited[nr][nc] and self.grid[nr, nc] != 1:
                    # Store parent for path reconstruction
                    self.parent[(nr, nc)] = current
                    
                    # Recursive call
                    if self._dfs(nr, nc, end):
                        return True
        
        return False
    
    def _reconstruct_path(self, start, end):
        """
        Reconstruct path from parent dictionary
        """
        path = []
        current = end
        while current != start:
            path.append(current)
            current = self.parent.get(current)
            if current is None:
                return [start]
        path.append(start)
        return path[::-1]