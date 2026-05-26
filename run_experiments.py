import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, accuracy_score, balanced_accuracy_score
from scipy import stats

# Import library modules
from optimization_lib.network import create_network, fitness_function, set_weights
from optimization_lib.ga import GeneticAlgorithm
from optimization_lib.pso import ParticleSwarmOptimization

# Create results folder
os.makedirs("results", exist_ok=True)

def load_data():
    df = pd.read_csv("parkinsons_preprocessed.csv")
    X = df.drop(columns=["status"]).values
    y = df["status"].values
    return X, y

def run_ga_operator_comparison(X, y):
    """
    Compares two GA configurations to evaluate the influence of operators:
    Config A (Base/Heuristic GA): Xavier/Uniform Init, BLX-0.5 Crossover, Gaussian Mutation, Tournament Selection
    Config B (Alternative GA): He/Normal Init, Arithmetic Crossover, Uniform Mutation, Roulette Wheel Selection
    """
    print("\n--- Running GA Operator Sensitivity Analysis ---")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    model = create_network(input_dim=X.shape[1], hidden_layer_sizes=(10, 5), random_state=42)
    
    # Config A
    ga_a = GeneticAlgorithm(
        model=model, X=X_train, y=y_train, pop_size=50, generations=50,
        crossover_prob=0.8, mutation_prob=0.25, mutation_gene_rate=0.15,
        init_method="uniform", selection_method="tournament",
        crossover_method="blx_alpha", mutation_method="gaussian",
        tournament_size=3, blx_alpha=0.5, mutation_scale=0.1,
        metric="f1_macro", random_state=42
    )
    
    # Config B
    ga_b = GeneticAlgorithm(
        model=model, X=X_train, y=y_train, pop_size=50, generations=50,
        crossover_prob=0.8, mutation_prob=0.25, mutation_gene_rate=0.15,
        init_method="normal", selection_method="roulette",
        crossover_method="arithmetic", mutation_method="uniform",
        mutation_scale=0.1, metric="f1_macro", random_state=42
    )
    
    print("Running GA Config A...")
    best_w_a, best_fit_a, fit_hist_a, mean_hist_a = ga_a.solve()
    test_f1_a = fitness_function(best_w_a, model, X_test, y_test, "f1_macro")
    
    print("Running GA Config B...")
    best_w_b, best_fit_b, fit_hist_b, mean_hist_b = ga_b.solve()
    test_f1_b = fitness_function(best_w_b, model, X_test, y_test, "f1_macro")
    
    print(f"Config A (BLX, Gaussian, Tourn) - Train Fitness: {best_fit_a:.4f}, Test F1-macro: {test_f1_a:.4f}")
    print(f"Config B (Arith, Uniform, Roule) - Train Fitness: {best_fit_b:.4f}, Test F1-macro: {test_f1_b:.4f}")
    
    # Plot sensitivity analysis convergence
    plt.figure(figsize=(10, 6))
    plt.plot(fit_hist_a, label="Config A (BLX Crossover, Gaussian Mutation, Tournament Selection)", color="#1e3a8a", linewidth=2)
    plt.plot(fit_hist_b, label="Config B (Arithmetic Crossover, Uniform Mutation, Roulette Wheel Selection)", color="#b91c1c", linewidth=2)
    plt.title("GA Operator Sensitivity Analysis (F1-macro vs Generations)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Training Fitness (F1-macro)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig("results/ga_sensitivity.png", dpi=300)
    plt.close()
    
    # Save statistics
    with open("results/ga_sensitivity_stats.txt", "w", encoding="utf-8") as f:
        f.write("GA OPERATOR COMPARISON RESULTS\n")
        f.write("=============================\n")
        f.write(f"Config A (Tournament, BLX-0.5, Gaussian, Uniform Init):\n")
        f.write(f"  Final Train Fitness (F1-macro): {best_fit_a:.6f}\n")
        f.write(f"  Test F1-macro Score:            {test_f1_a:.6f}\n\n")
        f.write(f"Config B (Roulette, Arithmetic, Uniform Mutation, Normal Init):\n")
        f.write(f"  Final Train Fitness (F1-macro): {best_fit_b:.6f}\n")
        f.write(f"  Test F1-macro Score:            {test_f1_b:.6f}\n")
        
    return ga_a # Return the best configured GA constructor logic (Config A)

def run_main_comparison(X, y):
    """
    Compares the optimized GA configuration against Particle Swarm Optimization (PSO)
    across 10 different random seeds/splits to ensure statistical validity.
    """
    print("\n--- Running Main Algorithm Comparison (GA vs PSO) over 10 runs ---")
    
    num_runs = 10
    ga_results = []
    pso_results = []
    
    ga_histories = []
    pso_histories = []
    
    # For saving best configurations
    best_ga_model_info = {"fit": -1, "weights": None, "X_test": None, "y_test": None, "seed": None}
    best_pso_model_info = {"fit": -1, "weights": None, "X_test": None, "y_test": None, "seed": None}
    
    for i in range(num_runs):
        seed = 42 + i
        print(f"Run {i+1}/{num_runs} (Seed: {seed})...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed
        )
        
        model = create_network(input_dim=X.shape[1], hidden_layer_sizes=(10, 5), random_state=seed)
        
        # --- Run GA ---
        ga = GeneticAlgorithm(
            model=model, X=X_train, y=y_train, pop_size=60, generations=80,
            crossover_prob=0.8, mutation_prob=0.25, mutation_gene_rate=0.15,
            init_method="uniform", selection_method="tournament",
            crossover_method="blx_alpha", mutation_method="gaussian",
            tournament_size=3, blx_alpha=0.5, mutation_scale=0.1,
            metric="f1_macro", random_state=seed
        )
        best_w_ga, best_fit_ga, fit_hist_ga, _ = ga.solve()
        
        # Evaluate GA
        test_f1_ga = fitness_function(best_w_ga, model, X_test, y_test, "f1_macro")
        test_acc_ga = fitness_function(best_w_ga, model, X_test, y_test, "accuracy")
        test_bacc_ga = fitness_function(best_w_ga, model, X_test, y_test, "balanced_accuracy")
        
        ga_results.append({
            "run": i+1, "train_f1": best_fit_ga, "test_f1": test_f1_ga, 
            "test_acc": test_acc_ga, "test_bacc": test_bacc_ga
        })
        ga_histories.append(fit_hist_ga)
        
        if test_f1_ga > best_ga_model_info["fit"]:
            best_ga_model_info.update({
                "fit": test_f1_ga, "weights": best_w_ga, 
                "X_test": X_test, "y_test": y_test, "seed": seed
            })
            
        # --- Run PSO ---
        pso = ParticleSwarmOptimization(
            model=model, X=X_train, y=y_train, num_particles=60, generations=80,
            w_start=0.9, w_end=0.4, c1=1.6, c2=1.6, v_max=0.5,
            pos_min=-2.0, pos_max=2.0, init_method="uniform",
            metric="f1_macro", random_state=seed
        )
        best_w_pso, best_fit_pso, fit_hist_pso, _ = pso.solve()
        
        # Evaluate PSO
        test_f1_pso = fitness_function(best_w_pso, model, X_test, y_test, "f1_macro")
        test_acc_pso = fitness_function(best_w_pso, model, X_test, y_test, "accuracy")
        test_bacc_pso = fitness_function(best_w_pso, model, X_test, y_test, "balanced_accuracy")
        
        pso_results.append({
            "run": i+1, "train_f1": best_fit_pso, "test_f1": test_f1_pso, 
            "test_acc": test_acc_pso, "test_bacc": test_bacc_pso
        })
        pso_histories.append(fit_hist_pso)
        
        if test_f1_pso > best_pso_model_info["fit"]:
            best_pso_model_info.update({
                "fit": test_f1_pso, "weights": best_w_pso, 
                "X_test": X_test, "y_test": y_test, "seed": seed
            })
            
    df_ga = pd.DataFrame(ga_results)
    df_pso = pd.DataFrame(pso_results)
    
    # Save comparison dataframes
    df_ga.to_csv("results/ga_results.csv", index=False)
    df_pso.to_csv("results/pso_results.csv", index=False)
    
    # Statistical tests
    # Relational t-test and Wilcoxon signed-rank test
    t_stat, t_pval = stats.ttest_rel(df_ga["test_f1"], df_pso["test_f1"])
    wilc_stat, wilc_pval = stats.wilcoxon(df_ga["test_f1"], df_pso["test_f1"])
    
    with open("results/statistical_results.txt", "w", encoding="utf-8") as f:
        f.write("GA VS PSO STATISTICAL COMPARISON\n")
        f.write("===============================\n\n")
        f.write("Genetic Algorithm Summary:\n")
        f.write(f"  Train F1-macro: {df_ga['train_f1'].mean():.4f} ± {df_ga['train_f1'].std():.4f}\n")
        f.write(f"  Test F1-macro:  {df_ga['test_f1'].mean():.4f} ± {df_ga['test_f1'].std():.4f}\n")
        f.write(f"  Test Accuracy:  {df_ga['test_acc'].mean():.4f} ± {df_ga['test_acc'].std():.4f}\n")
        f.write(f"  Test Bal. Acc:  {df_ga['test_bacc'].mean():.4f} ± {df_ga['test_bacc'].std():.4f}\n\n")
        
        f.write("Particle Swarm Optimization Summary:\n")
        f.write(f"  Train F1-macro: {df_pso['train_f1'].mean():.4f} ± {df_pso['train_f1'].std():.4f}\n")
        f.write(f"  Test F1-macro:  {df_pso['test_f1'].mean():.4f} ± {df_pso['test_f1'].std():.4f}\n")
        f.write(f"  Test Accuracy:  {df_pso['test_acc'].mean():.4f} ± {df_pso['test_acc'].std():.4f}\n")
        f.write(f"  Test Bal. Acc:  {df_pso['test_bacc'].mean():.4f} ± {df_pso['test_bacc'].std():.4f}\n\n")
        
        f.write("Hypothesis Testing (F1-macro):\n")
        f.write(f"  Paired t-test:   t-stat={t_stat:.4f}, p-value={t_pval:.6f}\n")
        f.write(f"  Wilcoxon test:   stat={wilc_stat:.4f}, p-value={wilc_pval:.6f}\n\n")
        
        if wilc_pval < 0.05:
            better = "PSO" if df_pso['test_f1'].mean() > df_ga['test_f1'].mean() else "GA"
            f.write(f"Conclusion: There is a statistically significant difference (p < 0.05). {better} performs better.\n")
        else:
            f.write("Conclusion: There is no statistically significant difference between GA and PSO performance (p >= 0.05).\n")

    print("\nStatistical results saved to results/statistical_results.txt")

    # Plot 1: Convergence History
    plt.figure(figsize=(10, 6))
    ga_mean_curve = np.mean(ga_histories, axis=0)
    pso_mean_curve = np.mean(pso_histories, axis=0)
    
    # Plot bounds (std deviation shadow)
    ga_std_curve = np.std(ga_histories, axis=0)
    pso_std_curve = np.std(pso_histories, axis=0)
    
    epochs = range(len(ga_mean_curve))
    
    plt.plot(epochs, ga_mean_curve, label="Genetic Algorithm (GA)", color="#2563eb", linewidth=2.5)
    plt.fill_between(epochs, ga_mean_curve - ga_std_curve, ga_mean_curve + ga_std_curve, color="#2563eb", alpha=0.15)
    
    plt.plot(epochs, pso_mean_curve, label="Particle Swarm Optimization (PSO)", color="#10b981", linewidth=2.5)
    plt.fill_between(epochs, pso_mean_curve - pso_std_curve, pso_mean_curve + pso_std_curve, color="#10b981", alpha=0.15)
    
    plt.title("Optimization Convergence Comparison (F1-macro)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Generation / Iteration", fontsize=12)
    plt.ylabel("Training Fitness (F1-macro)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/convergence_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Boxplot comparing F1-macro and Accuracy
    # Prepare data for boxplot
    plot_data = pd.DataFrame({
        "Score": list(df_ga["test_f1"]) + list(df_pso["test_f1"]) + list(df_ga["test_acc"]) + list(df_pso["test_acc"]),
        "Algorithm": ["GA"]*10 + ["PSO"]*10 + ["GA"]*10 + ["PSO"]*10,
        "Metric": ["F1-macro"]*20 + ["Accuracy"]*20
    })
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Metric", y="Score", hue="Algorithm", data=plot_data, palette=["#3b82f6", "#10b981"], width=0.6)
    plt.title("Algorithm Performance Metrics Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Performance Metric", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.ylim(0.4, 1.05)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig("results/performance_boxplot.png", dpi=300)
    plt.close()

    # Plot 3: Confusion Matrices and ROC Curves for best models
    # Set weights to best GA model
    best_ga_model = create_network(input_dim=X.shape[1], hidden_layer_sizes=(10, 5), random_state=best_ga_model_info["seed"])
    set_weights(best_ga_model, best_ga_model_info["weights"])
    ga_preds = best_ga_model.predict(best_ga_model_info["X_test"])
    ga_probs = best_ga_model.predict_proba(best_ga_model_info["X_test"])[:, 1]
    
    # Set weights to best PSO model
    best_pso_model = create_network(input_dim=X.shape[1], hidden_layer_sizes=(10, 5), random_state=best_pso_model_info["seed"])
    set_weights(best_pso_model, best_pso_model_info["weights"])
    pso_preds = best_pso_model.predict(best_pso_model_info["X_test"])
    pso_probs = best_pso_model.predict_proba(best_pso_model_info["X_test"])[:, 1]
    
    # Save Confusion Matrices side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_ga = confusion_matrix(best_ga_model_info["y_test"], ga_preds)
    sns.heatmap(cm_ga, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
                xticklabels=["Healthy", "Parkinson's"], yticklabels=["Healthy", "Parkinson's"])
    axes[0].set_title("Genetic Algorithm Confusion Matrix", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    
    cm_pso = confusion_matrix(best_pso_model_info["y_test"], pso_preds)
    sns.heatmap(cm_pso, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False,
                xticklabels=["Healthy", "Parkinson's"], yticklabels=["Healthy", "Parkinson's"])
    axes[1].set_title("PSO Confusion Matrix", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    
    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=300)
    plt.close()
    
    # Save ROC curves
    plt.figure(figsize=(8, 6))
    fpr_ga, tpr_ga, _ = roc_curve(best_ga_model_info["y_test"], ga_probs)
    roc_auc_ga = auc(fpr_ga, tpr_ga)
    
    fpr_pso, tpr_pso, _ = roc_curve(best_pso_model_info["y_test"], pso_probs)
    roc_auc_pso = auc(fpr_pso, tpr_pso)
    
    plt.plot(fpr_ga, tpr_ga, color="#2563eb", lw=2, label=f"GA ROC (AUC = {roc_auc_ga:.4f})")
    plt.plot(fpr_pso, tpr_pso, color="#10b981", lw=2, label=f"PSO ROC (AUC = {roc_auc_pso:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves of Best Models", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/roc_curves.png", dpi=300)
    plt.close()
    
    print("All comparison plots generated and saved in results/ folder.")

if __name__ == "__main__":
    X, y = load_data()
    run_ga_operator_comparison(X, y)
    run_main_comparison(X, y)
