import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AlgorithmPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.algorithms = ['Dijkstra', 'A*', 'BFS', 'Bellman-Ford']
        
    def extract_features(self, grid, start, end):
        """
        Extract features from grid for algorithm prediction
        Returns feature vector
        """
        rows, cols = grid.shape
        
        # Basic features
        grid_size = rows * cols
        obstacle_density = np.sum(grid == 1) / grid_size
        
        # Distance features
        start_end_distance = abs(start[0] - end[0]) + abs(start[1] - end[1])
        
        # Complexity features
        open_spaces = np.sum(grid == 0)
        connected_components = self.count_connected_components(grid)
        
        # Shape features
        aspect_ratio = rows / cols
        
        features = [
            grid_size,
            obstacle_density,
            start_end_distance,
            open_spaces / grid_size,
            connected_components,
            aspect_ratio,
            rows,
            cols
        ]
        
        return np.array(features).reshape(1, -1)
    
    def count_connected_components(self, grid):
        """Count number of connected components (excluding walls)"""
        from scipy import ndimage
        # Simple approximation
        return 1  # Simplified for now
    
    def train_demo(self):
        """Train a demo model with synthetic data"""
        # Generate synthetic training data
        X_train = []
        y_train = []
        
        # Create example grids with known best algorithms
        for _ in range(200):
            size = np.random.randint(10, 30)
            density = np.random.random()
            grid = (np.random.random((size, size)) < density).astype(int)
            
            start = (0, 0)
            end = (size-1, size-1)
            
            features = self.extract_features(grid, start, end).flatten()
            X_train.append(features)
            
            # Simple heuristic for demo
            if density < 0.2:
                y_train.append(1)  # A* for sparse grids
            elif density > 0.6:
                y_train.append(0)  # Dijkstra for dense grids
            else:
                y_train.append(2)  # BFS for medium density
        
        X_train = np.array(X_train)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Save model
        os.makedirs('ai_models', exist_ok=True)
        joblib.dump(self.model, 'ai_models/trained_model.pkl')
        joblib.dump(self.scaler, 'ai_models/scaler.pkl')
    
    def predict(self, grid, start, end):
        """Predict best algorithm for given grid"""
        if not self.is_trained:
            self.train_demo()
        
        features = self.extract_features(grid, start, end)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return {
            'algorithm': self.algorithms[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': {self.algorithms[i]: float(prob) 
                            for i, prob in enumerate(probabilities)}
        }
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            self.model = joblib.load('ai_models/trained_model.pkl')
            self.scaler = joblib.load('ai_models/scaler.pkl')
            self.is_trained = True
        except:
            self.train_demo()