import numpy as np
from agents_policies import Agent


class SocialRewardCalculator:
    """
    classe pour calculer les récompenses sociales basées sur la consommation
    et l'empathie entre agents
    """
    def __init__(self, nb_agents, alpha=0.5, beta=0.5):
        """
        initialise le calculateur de récompense sociale
        
        parameters:
        -----------
        nb_agents : int
            nombre d'agents dans l'environnement
        alpha : float
            pondération entre la satisfaction personnelle et l'empathie (0-1)
        beta : float
            pondération du dernier repas par rapport à l'historique (0-1)
        """
        self.nb_agents = nb_agents
        self.alpha = alpha  # balance entre soi et les autres
        self.beta = beta    # balance entre dernier repas et historique
        
    def calculate_personal_satisfaction(self, agent):
        """
        calcule la satisfaction personnelle d'un agent basée sur son historique de repas
        
        parameters:
        -----------
        agent : Agent
            l'agent dont on calcule la satisfaction
            
        returns:
        --------
        float
            score de satisfaction personnelle
        """
        # poids du dernier repas
        last_meal = 1 if agent.meal_history[-1] > 0 else 0
        
        # poids de l'historique récent
        history_weight = sum(agent.meal_history) / len(agent.meal_history)
        
        # combinaison pondérée
        satisfaction = self.beta * last_meal + (1 - self.beta) * history_weight
        
        return satisfaction
    
    def calculate_rewards(self, agents):
        """
        calcule les récompenses émotionnelles pour tous les agents
        
        parameters:
        -----------
        agents : list
            liste des agents
            
        returns:
        --------
        list
            liste des récompenses pour chaque agent
        """
        # calcul des satisfactions personnelles
        personal_satisfactions = [self.calculate_personal_satisfaction(agent) for agent in agents]
        
        # calcul des récompenses émotionnelles
        rewards = []
        for idx, satisfaction in enumerate(personal_satisfactions):
            # satisfaction personnelle
            own_satisfaction = satisfaction
            
            # satisfaction moyenne des autres agents (empathie)
            others_satisfaction = np.mean([s for i, s in enumerate(personal_satisfactions) if i != idx])
            
            # combinaison pondérée des deux composantes
            emotional_reward = self.alpha * own_satisfaction + (1 - self.alpha) * others_satisfaction
            rewards.append(emotional_reward)
        
        return rewards