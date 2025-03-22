import numpy as np
import matplotlib.pyplot as plt
from marl_simulation import MARLSimulation

def run_experiments(algorithms=None, alphas=None, betas=None, episodes=50, max_steps=30):
    """
    Exécute une série d'expériences avec différentes configurations
    
    Parameters:
    -----------
    algorithms : list
        Liste des algorithmes à tester
    alphas : list
        Liste des valeurs alpha à tester
    betas : list
        Liste des valeurs beta à tester
    episodes : int
        Nombre d'épisodes par expérience
    max_steps : int
        Nombre maximum d'étapes par épisode
        
    Returns:
    --------
    dict
        Résultats des expériences
    """
    if algorithms is None:
        algorithms = ["q_learning", "dqn"]
    if alphas is None:
        alphas = [0.3, 0.5, 0.7]
    if betas is None:
        betas = [0.5]
    
    results = {}
    
    for algorithm in algorithms:
        algorithm_results = {}
        
        for alpha in alphas:
            alpha_results = {}
            
            for beta in betas:
                print(f"\nDémarrage expérience: {algorithm}, alpha={alpha}, beta={beta}")
                
                # Création et exécution de la simulation
                simulation = MARLSimulation(
                    env_size=6,
                    nb_agents=3,
                    algorithm=algorithm,
                    episodes=episodes,
                    max_steps=max_steps,
                    alpha=alpha,
                    beta=beta
                )
                
                # Entraînement sans visualisation
                simulation.train(visualize_every=episodes+1)  # Jamais visualiser pendant l'expérience
                
                # Enregistrement des résultats
                alpha_results[beta] = {
                    'episode_rewards': simulation.episode_rewards,
                    'social_welfare': simulation.social_welfare
                }
                
                print(f"Expérience terminée: {algorithm}, alpha={alpha}, beta={beta}")
            
            algorithm_results[alpha] = alpha_results
        
        results[algorithm] = algorithm_results
    
    return results

def plot_comparison(results, metric='social_welfare'):
    """
    Visualise les résultats des expériences
    
    Parameters:
    -----------
    results : dict
        Résultats des expériences
    metric : str
        Métrique à visualiser ('episode_rewards' ou 'social_welfare')
    """
    plt.figure(figsize=(15, 10))
    algorithms = list(results.keys())
    alphas = list(results[algorithms[0]].keys())
    
    # Création d'une grille de sous-graphiques
    fig, axes = plt.subplots(len(algorithms), len(alphas), figsize=(15, 10), sharex=True)
    fig.suptitle(f'Comparaison de {metric}', fontsize=16)
    
    # Tracé des courbes
    for i, algorithm in enumerate(algorithms):
        for j, alpha in enumerate(alphas):
            ax = axes[i, j] if len(algorithms) > 1 and len(alphas) > 1 else axes[j] if len(algorithms) == 1 else axes[i]
            
            beta_results = results[algorithm][alpha]
            betas = list(beta_results.keys())
            
            for beta in betas:
                data = beta_results[beta][metric]
                x = np.arange(1, len(data) + 1)
                
                # Lissage pour une meilleure visualisation (moyenne mobile)
                window_size = max(1, len(data) // 10)
                smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
                
                ax.plot(x[window_size-1:], smoothed_data, label=f'Beta={beta}')
                ax.set_title(f'{algorithm}, Alpha={alpha}')
                ax.set_xlabel('Épisode')
                ax.set_ylabel(metric)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{metric}_comparison.png')
    plt.show()

def analyze_convergence(results):
    """
    Analyse la convergence des différents algorithmes
    
    Parameters:
    -----------
    results : dict
        Résultats des expériences
        
    Returns:
    --------
    dict
        Statistiques de convergence
    """
    convergence_stats = {}
    
    for algorithm in results:
        algorithm_stats = {}
        
        for alpha in results[algorithm]:
            alpha_stats = {}
            
            for beta in results[algorithm][alpha]:
                social_welfare = results[algorithm][alpha][beta]['social_welfare']
                
                # Calcul de la moyenne des 10 derniers épisodes
                final_welfare = np.mean(social_welfare[-10:])
                
                # Calcul de la volatilité (écart-type)
                volatility = np.std(social_welfare[-10:])
                
                # Détection de convergence (écart-type faible dans les derniers épisodes)
                is_converged = volatility < 0.1 * final_welfare  # Si l'écart-type est inférieur à 10% de la valeur finale
                
                alpha_stats[beta] = {
                    'final_welfare': final_welfare,
                    'volatility': volatility,
                    'is_converged': is_converged
                }
            
            algorithm_stats[alpha] = alpha_stats
        
        convergence_stats[algorithm] = algorithm_stats
    
    return convergence_stats

def print_convergence_stats(stats):
    """
    Affiche les statistiques de convergence
    
    Parameters:
    -----------
    stats : dict
        Statistiques de convergence
    """
    print("\n===== Statistiques de convergence =====")
    
    for algorithm in stats:
        print(f"\nAlgorithme: {algorithm}")
        
        for alpha in stats[algorithm]:
            print(f"  Alpha = {alpha}:")
            
            for beta in stats[algorithm][alpha]:
                s = stats[algorithm][alpha][beta]
                print(f"    Beta = {beta}: "
                      f"Bien-être final = {s['final_welfare']:.2f}, "
                      f"Volatilité = {s['volatility']:.2f}, "
                      f"Convergé = {s['is_converged']}")

if __name__ == "__main__":
    # Exécution des expériences
    print("Démarrage des expériences...")
    
    results = run_experiments(
        algorithms=["q_learning", "dqn"],
        alphas=[0.3, 0.5, 0.7],
        betas=[0.3, 0.7],
        episodes=50,
        max_steps=30
    )
    
    # Analyse des résultats
    print("\nAnalyse des résultats...")
    
    # Visualisation du bien-être social
    plot_comparison(results, metric='social_welfare')
    
    # Visualisation des récompenses par épisode
    plot_comparison(results, metric='episode_rewards')
    
    # Analyse de la convergence
    convergence_stats = analyze_convergence(results)
    print_convergence_stats(convergence_stats)
    
    print("\nAnalyse terminée!")