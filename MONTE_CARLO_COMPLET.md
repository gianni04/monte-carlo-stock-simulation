# 🎲 Monte Carlo — Cours Complet de A à Z

> **Objectif** : Comprendre la simulation Monte Carlo en finance, ses fondements mathématiques, et l'implémenter en Python pas à pas.

---

## 📚 Table des matières

1. [Qu'est-ce que Monte Carlo ?](#1--quest-ce-que-monte-carlo-)
2. [Les fondements mathématiques](#2--les-fondements-mathématiques)
3. [Le mouvement brownien géométrique (GBM)](#3--le-mouvement-brownien-géométrique-gbm)
4. [Application en finance](#4--application-en-finance)
5. [Code Python complet — étape par étape](#5--code-python-complet--étape-par-étape)
6. [Résultats et interprétation](#6--résultats-et-interprétation)
7. [Aller plus loin : VaR et pricing d'options](#7--aller-plus-loin--var-et-pricing-doptions)

---

## 1 — Qu'est-ce que Monte Carlo ?

La **simulation Monte Carlo** est une technique numérique qui utilise des **tirages aléatoires répétés** pour estimer des résultats probabilistes. Le nom vient du casino de Monte-Carlo à Monaco.

### Principe général
- On ne peut pas calculer analytiquement un résultat → trop complexe
- On **simule** des milliers/millions de scénarios aléatoires
- On **observe la distribution** des résultats pour inférer des probabilités

### En finance, on l'utilise pour :
| Problème | Application Monte Carlo |
|---|---|
| Prix futur d'une action | Simuler N trajectoires de prix |
| Valeur d'une option | Moyenne des payoffs actualisés |
| Value at Risk (VaR) | Distribution des pertes simulées |
| Portefeuille | Simuler corrélations et pertes |

---

## 2 — Les fondements mathématiques

### 2.1 La loi des grands nombres

Si on répète une expérience aléatoire un grand nombre de fois N :

```
E[X] ≈ (1/N) × Σ Xi   quand N → ∞
```

L'**espérance** d'une variable aléatoire est approximée par la **moyenne empirique** de ses réalisations.

### 2.2 Le théorème central limite

La somme de N variables aléatoires i.i.d. (indépendantes, identiquement distribuées) tend vers une **loi normale** :

```
√N × (X̄ - μ) / σ  →  N(0, 1)   quand N → ∞
```

Cela justifie pourquoi on utilise la loi normale pour modéliser les **rendements** financiers.

### 2.3 Les rendements logarithmiques

En finance, on modélise les **log-rendements** plutôt que les rendements simples :

```
r_t = ln(P_t / P_{t-1})
```

Propriétés importantes :
- Les log-rendements sont **additifs** dans le temps
- Ils suivent approximativement une **loi normale**
- Ils empêchent les prix de devenir négatifs

---

## 3 — Le mouvement brownien géométrique (GBM)

C'est le **modèle de référence** pour simuler les prix d'actifs financiers.

### 3.1 L'équation différentielle stochastique (EDS)

```
dS_t = μ × S_t × dt + σ × S_t × dW_t
```

Où :
- `S_t`  = Prix de l'actif au temps t
- `μ`    = Drift (tendance moyenne, taux de rendement annualisé)
- `σ`    = Volatilité (écart-type annualisé)
- `dW_t` = Incrément d'un mouvement brownien standard (processus de Wiener)
- `dt`   = Pas de temps infinitésimal

### 3.2 La solution analytique (formule de Black-Scholes)

Par le lemme d'Itô, la solution exacte est :

```
S_t = S_0 × exp( (μ - σ²/2) × t + σ × W_t )
```

Ou en forme discrète pour la simulation :

```
S_{t+Δt} = S_t × exp( (μ - σ²/2) × Δt + σ × √Δt × Z )
```

Où `Z ~ N(0,1)` est un tirage aléatoire standard.

### 3.3 Décomposition du terme

| Terme | Rôle |
|---|---|
| `(μ - σ²/2) × Δt` | Drift ajusté (correction d'Itô) |
| `σ × √Δt × Z` | Composante aléatoire (choc) |
| `exp(...)` | Garantit que S_t > 0 toujours |

> ⚠️ **Pourquoi `μ - σ²/2` et pas juste `μ` ?**
> C'est la **correction d'Itô**. Comme ln est une fonction concave, E[ln(X)] < ln(E[X]). Le terme `-σ²/2` corrige ce biais pour que E[S_t] = S_0 × exp(μ × t).

---

## 4 — Application en finance

### 4.1 Estimation des paramètres

À partir des données historiques :

```python
# Rendements logarithmiques journaliers
returns = np.log(prices / prices.shift(1)).dropna()

# Drift journalier
mu_daily = returns.mean()

# Volatilité journalière
sigma_daily = returns.std()

# Annualisation (252 jours de trading)
mu_annual    = mu_daily * 252
sigma_annual = sigma_daily * np.sqrt(252)
```

### 4.2 Pas de temps

Pour une simulation sur T jours avec N_steps pas :
```
Δt = T / N_steps      (en fraction d'année si μ et σ sont annualisés)
```

### 4.3 Nombre de simulations

| N simulations | Précision | Usage |
|---|---|---|
| 100 | Faible | Test rapide |
| 1 000 | Moyenne | Visualisation |
| 10 000 | Bonne | Analyse standard |
| 100 000+ | Excellente | Production / VaR précise |

---

## 5 — Code Python complet — étape par étape

```python
# ============================================================
# SIMULATION MONTE CARLO — CODE COMPLET
# Finance · L3 Économie/Finance
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
from scipy import stats

# ────────────────────────────────────────────────────────────
# ÉTAPE 1 : TÉLÉCHARGEMENT DES DONNÉES HISTORIQUES
# ────────────────────────────────────────────────────────────

ticker  = "TTE.PA"      # TotalEnergies (Euronext Paris)
start   = "2022-01-01"
end     = "2024-12-31"

data    = yf.download(ticker, start=start, end=end, progress=False)
prices  = data["Close"].dropna()

print(f"Données : {len(prices)} jours de cotation")
print(f"Prix initial  S₀ = {prices.iloc[0]:.2f} €")
print(f"Prix final    Sₙ = {prices.iloc[-1]:.2f} €")


# ────────────────────────────────────────────────────────────
# ÉTAPE 2 : CALCUL DES PARAMÈTRES (μ et σ)
# ────────────────────────────────────────────────────────────

# Log-rendements journaliers
log_returns = np.log(prices / prices.shift(1)).dropna()

mu_daily    = log_returns.mean()           # Drift journalier
sigma_daily = log_returns.std()            # Volatilité journalière

# Annualisation
TRADING_DAYS = 252
mu_annual    = mu_daily    * TRADING_DAYS
sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS)

print(f"\n📊 Paramètres estimés :")
print(f"   μ journalier   = {mu_daily:.6f}   ({mu_annual*100:.2f}% annualisé)")
print(f"   σ journalier   = {sigma_daily:.6f}  ({sigma_annual*100:.2f}% annualisé)")

# Test de normalité des rendements (Jarque-Bera)
jb_stat, jb_pval = stats.jarque_bera(log_returns)
print(f"\n   Jarque-Bera test : stat={jb_stat:.2f}, p-value={jb_pval:.4f}")
if jb_pval < 0.05:
    print("   ⚠️  Rendements non-normaux (queues épaisses possibles)")
else:
    print("   ✅ Hypothèse de normalité non rejetée")


# ────────────────────────────────────────────────────────────
# ÉTAPE 3 : PARAMÈTRES DE SIMULATION
# ────────────────────────────────────────────────────────────

S0          = float(prices.iloc[-1])    # Prix de départ = dernier prix connu
T           = 252                       # Horizon : 1 an de trading
N_simul     = 10_000                   # Nombre de simulations
N_steps     = T                         # Pas journaliers
dt          = 1 / TRADING_DAYS          # Δt en fraction d'année

print(f"\n🎲 Paramètres Monte Carlo :")
print(f"   S₀          = {S0:.2f} €")
print(f"   T           = {T} jours ({T/TRADING_DAYS:.1f} an)")
print(f"   Simulations = {N_simul:,}")
print(f"   Δt          = {dt:.6f} an")


# ────────────────────────────────────────────────────────────
# ÉTAPE 4 : LA SIMULATION (cœur de Monte Carlo)
# ────────────────────────────────────────────────────────────
# Formule : S_{t+Δt} = S_t × exp( (μ - σ²/2)×Δt + σ×√Δt×Z )
# Z ~ N(0,1) tiré aléatoirement

np.random.seed(42)   # Reproductibilité

# Matrice des chocs aléatoires : shape (N_steps, N_simul)
Z = np.random.standard_normal((N_steps, N_simul))

# Incrément log-prix à chaque pas
drift     = (mu_annual - 0.5 * sigma_annual**2) * dt
diffusion = sigma_annual * np.sqrt(dt) * Z

# Variations cumulées (log-prix relatifs)
log_increments = drift + diffusion                     # (N_steps, N_simul)
cum_log        = np.cumsum(log_increments, axis=0)     # cumulé dans le temps

# Prix simulés : S0 × exp(cumul des log-incréments)
# On ajoute S0 comme ligne initiale
price_paths       = S0 * np.exp(cum_log)               # (N_steps, N_simul)
price_paths_full  = np.vstack([np.full(N_simul, S0), price_paths])  # (N_steps+1, N_simul)

print(f"\n✅ Simulation terminée ! Matrice de prix : {price_paths_full.shape}")


# ────────────────────────────────────────────────────────────
# ÉTAPE 5 : ANALYSE DES PRIX FINAUX
# ────────────────────────────────────────────────────────────

final_prices = price_paths_full[-1, :]     # Prix à la fin de la simulation

# Statistiques descriptives
mean_price   = np.mean(final_prices)
median_price = np.median(final_prices)
std_price    = np.std(final_prices)
min_price    = np.min(final_prices)
max_price    = np.max(final_prices)

# Intervalles de confiance à 95%
ci_low  = np.percentile(final_prices, 2.5)
ci_high = np.percentile(final_prices, 97.5)

# VaR à 95% (perte potentielle)
VaR_95 = np.percentile(final_prices, 5)
VaR_99 = np.percentile(final_prices, 1)

print(f"\n📈 Résultats — Distribution des prix dans {T} jours :")
print(f"   Prix initial (S₀)          = {S0:.2f} €")
print(f"   Prix moyen simulé (E[S_T]) = {mean_price:.2f} €")
print(f"   Médiane                    = {median_price:.2f} €")
print(f"   Écart-type                 = {std_price:.2f} €")
print(f"   Min / Max                  = {min_price:.2f} € / {max_price:.2f} €")
print(f"\n   Intervalle de confiance 95% : [{ci_low:.2f} € ; {ci_high:.2f} €]")
print(f"   VaR 95% (floor price)      = {VaR_95:.2f} €")
print(f"   VaR 99% (floor price)      = {VaR_99:.2f} €")

prob_gain = np.mean(final_prices > S0)
print(f"\n   Probabilité de gain (S_T > S₀) = {prob_gain*100:.1f}%")
print(f"   Probabilité de perte           = {(1-prob_gain)*100:.1f}%")


# ────────────────────────────────────────────────────────────
# ÉTAPE 6 : VISUALISATIONS
# ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Monte Carlo — {ticker} | {N_simul:,} simulations | Horizon {T} jours",
             fontsize=16, fontweight='bold')

# ── Graphique 1 : Trajectoires simulées ──
ax1 = axes[0, 0]
ax1.plot(price_paths_full[:, :200], alpha=0.05, color='steelblue', linewidth=0.5)
ax1.axhline(S0, color='black', linestyle='--', linewidth=1.5, label=f'S₀ = {S0:.2f}€')
ax1.axhline(mean_price, color='red', linestyle='-', linewidth=1.5, label=f'E[S_T] = {mean_price:.2f}€')
ax1.fill_between(range(N_steps+1),
                 np.percentile(price_paths_full, 5, axis=1),
                 np.percentile(price_paths_full, 95, axis=1),
                 alpha=0.2, color='orange', label='IC 90%')
ax1.set_title('Trajectoires simulées (200 sur 10 000)')
ax1.set_xlabel('Jours')
ax1.set_ylabel('Prix (€)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Graphique 2 : Distribution des prix finaux ──
ax2 = axes[0, 1]
ax2.hist(final_prices, bins=100, color='steelblue', edgecolor='white',
         alpha=0.8, density=True)
ax2.axvline(S0,           color='black', linestyle='--', linewidth=2, label=f'S₀ = {S0:.2f}€')
ax2.axvline(mean_price,   color='red',   linestyle='-',  linewidth=2, label=f'Moyenne = {mean_price:.2f}€')
ax2.axvline(ci_low,       color='orange',linestyle=':',  linewidth=1.5, label=f'IC 95%: [{ci_low:.0f}€ ; {ci_high:.0f}€]')
ax2.axvline(ci_high,      color='orange',linestyle=':',  linewidth=1.5)
ax2.axvline(VaR_95,       color='darkred',linestyle='-.',linewidth=2, label=f'VaR 95% = {VaR_95:.2f}€')
ax2.set_title(f'Distribution des prix finaux S_T')
ax2.set_xlabel('Prix final (€)')
ax2.set_ylabel('Densité')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Graphique 3 : Log-rendements historiques ──
ax3 = axes[1, 0]
ax3.hist(log_returns, bins=60, color='teal', edgecolor='white', alpha=0.8, density=True)
x_range = np.linspace(log_returns.min(), log_returns.max(), 200)
normal_fit = stats.norm.pdf(x_range, mu_daily, sigma_daily)
ax3.plot(x_range, normal_fit, 'r-', linewidth=2, label='Loi normale ajustée')
ax3.set_title('Distribution des log-rendements historiques')
ax3.set_xlabel('Log-rendement journalier')
ax3.set_ylabel('Densité')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── Graphique 4 : Convergence (loi des grands nombres) ──
ax4 = axes[1, 1]
sample_sizes = np.logspace(1, np.log10(N_simul), 100, dtype=int)
running_means = [np.mean(final_prices[:n]) for n in sample_sizes]
ax4.semilogx(sample_sizes, running_means, color='purple', linewidth=2)
ax4.axhline(mean_price, color='red', linestyle='--', linewidth=1.5,
            label=f'Moyenne finale = {mean_price:.2f}€')
ax4.set_title('Convergence (Loi des grands nombres)')
ax4.set_xlabel('Nombre de simulations (échelle log)')
ax4.set_ylabel('Moyenne convergée (€)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('monte_carlo_resultats.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n💾 Graphique sauvegardé : monte_carlo_resultats.png")


# ────────────────────────────────────────────────────────────
# ÉTAPE 7 : PRICING D'UNE OPTION CALL (BONUS)
# ────────────────────────────────────────────────────────────
# Par Monte Carlo : C = e^{-rT} × E[max(S_T - K, 0)]

K   = S0 * 1.05    # Strike = prix actuel +5% (option légèrement OTM)
r   = 0.035        # Taux sans risque (OAT 10 ans ~3.5%)
T_y = T / TRADING_DAYS  # Horizon en années

payoffs      = np.maximum(final_prices - K, 0)
call_price   = np.exp(-r * T_y) * np.mean(payoffs)
call_std_err = np.exp(-r * T_y) * np.std(payoffs) / np.sqrt(N_simul)

print(f"\n💡 Pricing d'option Call (Monte Carlo) :")
print(f"   Strike K      = {K:.2f} €  (+5% OTM)")
print(f"   Taux sans risque r = {r*100:.1f}%")
print(f"   Prix du Call  = {call_price:.4f} €")
print(f"   Erreur std    = ±{call_std_err:.4f} €")
print(f"   IC 95% Call   = [{call_price - 1.96*call_std_err:.4f} € ; {call_price + 1.96*call_std_err:.4f} €]")
```

---

## 6 — Résultats et interprétation

### Ce que tu lis sur les graphiques

**Graphique 1 — Trajectoires** : Chaque ligne bleue = 1 scénario futur possible. La zone orange = l'intervalle de confiance à 90% des trajectoires. Plus l'horizon est long, plus l'éventail s'élargit (incertitude croissante).

**Graphique 2 — Distribution finale** : Forme de cloche asymétrique (log-normale). La ligne rouge = espérance. La zone à gauche du VaR 95% = les 5% des pires scénarios.

**Graphique 3 — Rendements historiques** : Si la courbe rouge (normale) colle bien à l'histogramme → GBM justifié. Si les queues sont épaisses → attention, le modèle sous-estime les extrêmes.

**Graphique 4 — Convergence** : Illustre la **loi des grands nombres**. La moyenne des prix finaux converge vers E[S_T] quand N → ∞. En pratique, 10 000 simulations suffisent.

### Interprétation économique

```
E[S_T] = S₀ × exp(μ × T)    (en théorie)
```

Si `μ > 0` → Le marché anticipe une tendance haussière. La moyenne des prix simulés sera supérieure à S₀. Mais la **médiane** sera toujours ≤ à la moyenne (distribution log-normale asymétrique).

---

## 7 — Aller plus loin : VaR et pricing d'options

### 7.1 Value at Risk Monte Carlo

```python
# VaR à horizon 1 jour, niveau 99%
rendements_sim   = np.log(final_prices / S0)          # Rendements simulés sur T jours
VaR_1j_99        = np.percentile(rendements_sim, 1)   # 1er percentile
perte_max_1j     = S0 * (1 - np.exp(VaR_1j_99))
print(f"VaR 99% sur {T} jours = {perte_max_1j:.2f} €")
```

### 7.2 Pricing d'option — Formule complète

```
C_Monte_Carlo = e^{-rT} × (1/N) × Σ max(S_T^i - K, 0)
```

Pour une option **Put** :
```
P_Monte_Carlo = e^{-rT} × (1/N) × Σ max(K - S_T^i, 0)
```

### 7.3 Limites du modèle GBM

| Limite | Effet en pratique |
|---|---|
| Rendements supposés normaux | Sous-estime les crashs (queues épaisses) |
| μ et σ constants dans le temps | Faux en période de crise |
| Pas de sauts (discontinuités) | Manque les chocs brutaux type COVID-19 |
| Corrélations entre actifs non modélisées | Problème pour les portefeuilles |

**Solutions alternatives** :
- Modèle de Heston (volatilité stochastique)
- Modèle à sauts de Merton (Jump-Diffusion)
- Copules pour les corrélations (portefeuille)

---

## 📌 Résumé des formules clés

| Formule | Description |
|---|---|
| `r_t = ln(P_t / P_{t-1})` | Log-rendement |
| `S_{t+Δt} = S_t × exp((μ - σ²/2)Δt + σ√Δt × Z)` | GBM discrétisé |
| `μ_ann = μ_day × 252` | Annualisation drift |
| `σ_ann = σ_day × √252` | Annualisation volatilité |
| `C = e^{-rT} × E[max(S_T - K, 0)]` | Prix option Call |
| `VaR_α = -Q_α(P&L)` | Value at Risk |

---

*Fichier généré le 07/05/2026 — Cours L3 Finance · Monte Carlo de A à Z*
