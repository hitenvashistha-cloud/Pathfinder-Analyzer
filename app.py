import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from collections import defaultdict

# Import algorithms
from algorithms.dijkstra import Dijkstra
from algorithms.a_star import AStar
from algorithms.bfs import BFS
from algorithms.bellman_ford import BellmanFord

# Import AI model
from ai_models.algorithm_predictor import AlgorithmPredictor

# Import utilities
from utils.grid_utils import *

# Page configuration
st.set_page_config(
    page_title="AI Pathfinding Visualizer",
    page_icon="🤖",
    layout="wide"
)

if 'grid' not in st.session_state:
    st.session_state.grid = create_empty_grid(15, 15)
if 'start' not in st.session_state:
    st.session_state.start = (0, 0)
if 'end' not in st.session_state:
    st.session_state.end = (14, 14)
if 'draw_mode' not in st.session_state:
    st.session_state.draw_mode = 'wall'  # wall, start, end
if 'ai_predictor' not in st.session_state:
    st.session_state.ai_predictor = AlgorithmPredictor()

st.title("🤖  Pathfinding Visualizer")
st.markdown("""
    This application demonstrates various pathfinding algorithms with AI-powered 
    algorithm selection. Click on cells to add walls, then run algorithms to find 
    the shortest path!
""")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Algorithm selection
    algorithm = st.selectbox(
        "Select Algorithm",
        ["Dijkstra", "A*", "BFS", "Bellman-Ford", "🤖 AI Recommended"]
    )
    
    st.divider()
    
    # Grid configuration
    st.subheader("Grid Settings")
    grid_size = st.slider("Grid Size", 5, 30, 15)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Grid"):
            st.session_state.grid = create_empty_grid(grid_size, grid_size)
            st.session_state.start = (0, 0)
            st.session_state.end = (grid_size-1, grid_size-1)
            st.rerun()
    
    with col2:
        if st.button("🎲 Random Maze"):
            st.session_state.grid = create_empty_grid(grid_size, grid_size)
            st.session_state.grid = generate_random_maze(st.session_state.grid, density=0.3)
            st.rerun()
    
    st.divider()
    
    # Draw mode selection
    st.subheader("Draw Mode")
    draw_mode = st.radio(
        "Click on cells to:",
        ["🚧 Add Walls", "🏁 Set Start", "🏆 Set End"],
        horizontal=True
    )
    
    if draw_mode == "🚧 Add Walls":
        st.session_state.draw_mode = 'wall'
    elif draw_mode == "🏁 Set Start":
        st.session_state.draw_mode = 'start'
    else:
        st.session_state.draw_mode = 'end'
    
    st.divider()
    
    # Performance metrics display
    st.subheader("📊 Performance Metrics")
    metrics_placeholder = st.empty()

# Main area - create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Grid Visualization")
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    rows, cols = st.session_state.grid.shape
    cell_size = 1.0
    
    # Draw grid cells
    for i in range(rows):
        for j in range(cols):
            color = 'white'
            if st.session_state.grid[i, j] == 1:  # Wall
                color = 'black'
            elif (i, j) == st.session_state.start:
                color = 'green'
            elif (i, j) == st.session_state.end:
                color = 'red'
            
            rect = Rectangle((j, rows-1-i), 1, 1, facecolor=color, edgecolor='gray')
            ax.add_patch(rect)
    
    # Set grid properties
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Display the grid
    grid_display = st.pyplot(fig)
    
    # Handle cell clicks (using session state and rerun)
    # Note: Streamlit doesn't support direct click events on matplotlib
    # We'll use buttons for cell selection instead
    
    # Alternative: Create clickable grid using buttons
    st.subheader("Click to Edit Grid")
    cols_per_row = 8
    for i in range(rows):
        button_cols = st.columns(cols_per_row)
        for j in range(cols):
            col_idx = j % cols_per_row
            cell_value = st.session_state.grid[i, j]
            
            if (i, j) == st.session_state.start:
                label = "🏁"
            elif (i, j) == st.session_state.end:
                label = "🏆"
            elif cell_value == 1:
                label = "🧱"
            else:
                label = "⬜"
            
            if button_cols[col_idx].button(label, key=f"cell_{i}_{j}", use_container_width=True):
                if st.session_state.draw_mode == 'wall':
                    st.session_state.grid[i, j] = 1 if cell_value != 1 else 0
                elif st.session_state.draw_mode == 'start':
                    st.session_state.start = (i, j)
                elif st.session_state.draw_mode == 'end':
                    st.session_state.end = (i, j)
                st.rerun()

with col2:
    st.subheader(" Run Algorithms")
    
    # Run button
    if st.button("Run Algorithm", type="primary", use_container_width=True):
        # Ensure grid size matches start/end
        if st.session_state.start[0] >= rows or st.session_state.start[1] >= cols:
            st.session_state.start = (0, 0)
        if st.session_state.end[0] >= rows or st.session_state.end[1] >= cols:
            st.session_state.end = (rows-1, cols-1)
        
        # Check if start and end are not walls
        if st.session_state.grid[st.session_state.start] == 1:
            st.error(" Start position is a wall! Please set start on empty cell.")
        elif st.session_state.grid[st.session_state.end] == 1:
            st.error(" End position is a wall! Please set end on empty cell.")
        else:
            # Select algorithm
            if algorithm == " AI Recommended":
                with st.spinner(" AI analyzing grid..."):
                    prediction = st.session_state.ai_predictor.predict(
                        st.session_state.grid, 
                        st.session_state.start, 
                        st.session_state.end
                    )
                    st.info(f" AI recommends: **{prediction['algorithm']}** (Confidence: {prediction['confidence']:.1%})")
                    
                    # Show probability distribution
                    st.write("**Algorithm probabilities:**")
                    for algo, prob in prediction['probabilities'].items():
                        st.progress(prob, text=f"{algo}: {prob:.1%}")
                    
                    selected_algo = prediction['algorithm']
            else:
                selected_algo = algorithm
            
            # Run selected algorithm
            with st.spinner(f"Running {selected_algo}..."):
                start_time = time.time()
                
                if selected_algo == "Dijkstra":
                    algo = Dijkstra(st.session_state.grid)
                    path, visited, distance = algo.find_path(st.session_state.start, st.session_state.end)
                elif selected_algo == "A*":
                    algo = AStar(st.session_state.grid)
                    path, visited, distance = algo.find_path(st.session_state.start, st.session_state.end)
                elif selected_algo == "BFS":
                    algo = BFS(st.session_state.grid)
                    path, visited, distance = algo.find_path(st.session_state.start, st.session_state.end)
                elif selected_algo == "Bellman-Ford":
                    algo = BellmanFord(st.session_state.grid)
                    path, visited, distance = algo.find_path(st.session_state.start, st.session_state.end)
                else:
                    st.error("Unknown algorithm")
                    path = None
                    visited = []
                    distance = float('inf')
                
                end_time = time.time()
                execution_time = end_time - start_time
            
            # Display results
            if path:
                st.success(f"Path found! Length: {distance} steps")
                st.metric("Path Length", distance)
                st.metric("Execution Time", f"{execution_time:.4f} seconds")
                st.metric("Nodes Visited", len(visited))
                
                # Show complexity
                st.write("**Time Complexity:**")
                if selected_algo == "Dijkstra":
                    st.code("O((V + E) log V) where V = vertices, E = edges")
                elif selected_algo == "A*":
                    st.code("O(b^d) where b = branching factor, d = depth")
                elif selected_algo == "BFS":
                    st.code("O(V + E) for unweighted graphs")
                elif selected_algo == "Bellman-Ford":
                    st.code("O(VE) where V = vertices, E = edges")
                
                # Visualize path in a new figure
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                for i in range(rows):
                    for j in range(cols):
                        color = 'white'
                        if st.session_state.grid[i, j] == 1:
                            color = 'black'
                        elif (i, j) == st.session_state.start:
                            color = 'green'
                        elif (i, j) == st.session_state.end:
                            color = 'red'
                        elif path and (i, j) in path:
                            color = 'yellow'
                        elif (i, j) in visited:
                            color = 'lightblue'
                        
                        rect = Rectangle((j, rows-1-i), 1, 1, facecolor=color, edgecolor='gray')
                        ax2.add_patch(rect)
                
                ax2.set_xlim(0, cols)
                ax2.set_ylim(0, rows)
                ax2.set_aspect('equal')
                ax2.axis('off')
                ax2.set_title(f"{selected_algo} - Path Visualization (Yellow = Path, Light Blue = Visited)")
                
                st.pyplot(fig2)
            else:
                st.error("❌ No path exists from start to end!")
    
    st.divider()
    
    # Additional options
    st.subheader("Advanced Options")
    if st.button("Compare All Algorithms", use_container_width=True):
        st.info("Comparing algorithm performance...")
        
        algorithms_to_test = {
            'Dijkstra': Dijkstra(st.session_state.grid),
            'A*': AStar(st.session_state.grid),
            'BFS': BFS(st.session_state.grid),
            'Bellman-Ford': BellmanFord(st.session_state.grid)
        }
        
        results = []
        for name, algo in algorithms_to_test.items():
            start_time = time.time()
            path, visited, distance = algo.find_path(st.session_state.start, st.session_state.end)
            exec_time = time.time() - start_time
            
            results.append({
                'Algorithm': name,
                'Path Length': len(path) - 1 if path else 'No path',
                'Nodes Visited': len(visited),
                'Time (s)': f"{exec_time:.4f}",
                'Path Found': 'Path exist' if path else 'Path not exist'
            })
        
        st.table(results)

# Footer
st.divider()
st.markdown("""
    **Educational Project** | Algorithms: Dijkstra, A*, BFS, Bellman-Ford | 
    ** AI Features**: Algorithm recommendation using Random Forest Classifier
""")