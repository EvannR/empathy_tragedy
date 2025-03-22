import numpy as np
from collections import deque

class Agent:
    def __init__(self, agent_id, position, memory_size=10):
        self.agent_id = agent_id
        self.position = position
        self.memory_size = memory_size
        self.meal_history = deque([0] * memory_size, maxlen=memory_size)
        self.total_meals = 0
    
    def record_meal(self, has_eaten, reward_value=0):
        """Enregistre si l'agent a mangé (1) ou non (0) à ce tour"""
        self.meal_history.append(1 if has_eaten else 0)
        if has_eaten:
            self.total_meals += 1
    
    def get_recent_meals(self):
        """Retourne le nombre de repas dans l'historique"""
        return sum(self.meal_history)
    
    def update_position(self, new_position):
        """Met à jour la position de l'agent"""
        self.position = new_position

