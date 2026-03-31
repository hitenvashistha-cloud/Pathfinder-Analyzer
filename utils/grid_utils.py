import numpy as np

def create_empty_grid(rows, cols):
    return np.zeros((rows, cols))

def add_wall(grid, row, col):
    grid[row, col] = 1
    return grid

def remove_wall(grid, row, col):
    grid[row, col] = 0
    return grid

def generate_random_maze(grid, density=0.3):
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            if np.random.random() < density:
                grid[i, j] = 1
    return grid

def generate_maze_pattern(grid, pattern='recursive_backtracking'):
    rows, cols = grid.shape
    if pattern == 'simple':
        # Simple vertical/horizontal lines
        for i in range(rows):
            grid[i, cols//2] = 1
        for j in range(cols):
            grid[rows//2, j] = 1
    elif pattern == 'spiral':
        # Spiral pattern (simplified)
        for i in range(min(rows, cols)//4):
            grid[i, i:cols-i] = 1
            grid[rows-1-i, i:cols-i] = 1
            grid[i:rows-i, i] = 1
            grid[i:rows-i, cols-1-i] = 1
    
    return grid

def validate_path_exists(grid, start, end):
    from collections import deque
    
    rows, cols = grid.shape
    visited = set([start])
    queue = deque([start])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        row, col = queue.popleft()
        if (row, col) == end:
            return True
        
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] != 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    return False