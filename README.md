# empathy_tragedy

#  Simulation Multi-Agent avec tentative d'empathie

!! attention !! les fichiers du dossier n'apparaissant ici sont des fichiers test ou passés qui ne servent plus à l'éxcution des simulations 

## Vue d'Ensemble

Le projet implémente un système d'apprentissage par renforcement multi-agent (MARL) où plusieurs agents interagissent dans un environnement de type grille pour collecter des ressources. La particularité est que la récompense de chaque agent dépend non seulement de sa propre consommation (satisfaction personnelle) mais aussi de celle des autres agents (empathie). 

L'architecture est modulaire et se compose de plusieurs fichiers:

1. `env.py` - Définit l'environnement
2. `agents.py` - Définit la structure des agents
3. `policies.py` - Implémente les algorithmes d'apprentissage (Q-Learning et DQN)
4. `marl_simulation.py` - Orchestre la simulation
5. `analyze_results.py` - Analyse et visualise les résultats

## 1. Environnement (`env.py`)

L'environnement est une grille 2D contenant des agents et des ressources:

- **`GridMaze`** (classe de base):
  - Définit une grille de taille configurable
  - Gère le déplacement des agents (UP, DOWN, LEFT, RIGHT, EXPLOIT)
  - Suit les positions des agents et les récompenses

- **`RandomizedGridMaze`** (classe dérivée):
  - Ajoute des ressources aléatoires dans la grille
  - Configure le comportement des ressources (apparition, consommation)
  - Permet de choisir si les ressources sont consommées automatiquement ou seulement via l'action EXPLOIT

L'environnement maintient l'état du système à chaque pas de temps et peut être réinitialisé pour un nouvel épisode.

## 2. Agents (`agents.py`)

La classe `Agent` encapsule les caractéristiques et comportements fondamentaux des agents:

- Stocke la position courante de l'agent
- Maintient un historique des repas sous forme d'une file circulaire (`deque`)
- Compte le nombre total de repas
- Fournit des méthodes pour enregistrer de nouveaux repas et calculer des statistiques

Chaque agent peut avoir une capacité de mémoire différente, ce qui affecte la longueur de son historique de repas.

## 3. Politiques d'Apprentissage (`policies.py`)

Ce fichier contient les algorithmes d'apprentissage par renforcement:

- **`QAgent`**: Implémentation du Q-Learning classique
  - Utilise une table pour stocker les valeurs Q
  - Utilise une politique epsilon-greedy pour l'exploration/exploitation
  - Met à jour les valeurs Q basées sur l'équation de Bellman

- **`DQNAgent`**: Implémentation du Deep Q-Network
  - Utilise des réseaux de neurones pour approximer la fonction Q
  - Implémente un buffer de replay pour l'apprentissage par lots
  - Utilise un réseau cible pour stabiliser l'apprentissage

- **`EmotionalModel`** et **`SocialRewardCalculator`**: 
  - Calculent les récompenses basées sur l'empathie
  - Pondèrent entre la satisfaction personnelle et celle des autres agents
  - Prennent en compte l'historique des repas et le dernier repas

- **`ReplayBuffer`**: 
  - Stocke les expériences (état, action, récompense, état suivant, terminal)
  - Échantillonne des lots d'expériences pour l'apprentissage

## 4. Simulation Principale (`marl_simulation.py`)

Ce fichier orchestre tout le processus d'apprentissage:

- Initialise l'environnement et les agents
- Crée des agents RL (Q-Learning ou DQN) selon le choix de l'algorithme
- Exécute des épisodes complets avec les étapes:
  1. Réinitialisation de l'environnement
  2. Sélection d'actions par les agents
  3. Exécution des actions dans l'environnement
  4. Calcul des récompenses sociales en fin d'épisode
  5. Apprentissage des agents
- Collecte des statistiques pour analyse
- Visualise l'environnement et les performances des agents

La méthode `get_state_representation` convertit l'état de la grille en une représentation adaptée à l'apprentissage par renforcement: position normalisée de l'agent et informations sur les ressources environnantes.

## 5. Analyse des Résultats (`analyze_results.py`)

Ce fichier fournit des outils pour:
- Exécuter des expériences avec différentes configurations (algorithmes, paramètres)
- Visualiser les courbes d'apprentissage
- Analyser la convergence des différents algorithmes
- Comparer les performances en termes de bien-être social et de récompenses individuelles

## Aspects Techniques Clés

### 1. Représentation d'État

Les états sont représentés comme des vecteurs de caractéristiques incluant:
- Position normalisée de l'agent (i, j divisés par la taille de la grille)
- Valeurs des ressources dans les 8 directions autour de l'agent

### 2. Modèle d'Empathie

Le modèle d'empathie est paramétré par deux coefficients:
- `alpha`: Équilibre entre satisfaction personnelle (1.0) et empathie (0.0)
- `beta`: Pondération entre le dernier repas et l'historique complet

### 3. Système de Récompense

À la fin de chaque épisode, la récompense finale comprend:
- Satisfaction personnelle = `beta` × (dernier repas) + (1-`beta`) × (moyenne de l'historique)
- Récompense émotionnelle = `alpha` × (satisfaction personnelle) + (1-`alpha`) × (moyenne des satisfactions des autres)

### 4. Gestion des Erreurs et Débogage

Le code inclut un système robuste de gestion des erreurs pour:
- Tracer les problèmes liés aux types de données et aux conversions
- Afficher des informations détaillées sur les états, actions et récompenses
- Continuer l'apprentissage même en cas d'erreur dans un épisode

## Paramétrisation et Flexibilité

Le système est hautement configurable:
- Taille de l'environnement
- Nombre d'agents
- Densité et dynamique des ressources
- Algorithme d'apprentissage (Q-Learning vs DQN)
- Paramètres d'empathie (alpha, beta)
- Hyperparamètres d'apprentissage (taux d'apprentissage, facteur d'actualisation, exploration)

Cette flexibilité permet de tester diverses hypothèses et configurations pour trouver les combinaisons optimales pour votre recherche MARL.

## Conclusion
