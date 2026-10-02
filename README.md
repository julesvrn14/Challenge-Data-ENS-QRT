# Challenge-Data-ENS-QRT
# Prédiction du signe de performance d'allocations d'actifs

## Contexte
Challenge de machine learning appliqué à la finance quantitative : prédire si une allocation d'actifs
donnée (un portefeuille construit selon une stratégie systématique) va générer un rendement positif ou
négatif lors de la prochaine session de trading, à partir de son comportement sur les 20 jours précédents
(rendements, volumes signés, turnover).

## Démarche

**Feature engineering** — au-delà des rendements bruts fournis, construction de features de momentum et
volatilité à plusieurs horizons, d'un effet marché global (moyenne cross-sectionnelle du jour), de ratios
performance/risque (type Sharpe), de mesures de liquidité (corrélation rendement/volume), et de features de
forme de distribution (skewness, kurtosis, max drawdown, EWMA).

**Validation rigoureuse** — cross-validation par date (et non par ligne) pour éviter toute fuite
d'information entre observations du même jour, utilisée de façon identique sur tous les modèles pour
garantir des comparaisons équitables.

**Test d'hypothèse rejeté proprement** — une feature catégorielle (groupe d'allocations) a été testée selon
trois approches (encodage brut, agrégats de groupe, rang relatif). Un test statistique apparié a montré
qu'elle n'apportait aucun gain significatif sur 3 modèles différents, menant à son abandon plutôt qu'à son
maintien par défaut.

**Modélisation** — comparaison de LightGBM, CatBoost, XGBoost et de modèles linéaires (Logistic Regression,
ElasticNet) contre des baselines naïves de référence (persistance, momentum). Tuning d'hyperparamètres par
grid search, sélection de features par permutation importance, et stacking final (méta-modèle apprenant à
combiner les prédictions des 3 modèles d'arbres) avec calibration du seuil de décision.

## Résultat
Gain confirmé sur le leaderboard public par rapport au benchmark de référence, dans un contexte où le signal
prédictif est intrinsèquement faible (caractéristique des séries financières) — l'accent a été mis sur la
rigueur méthodologique (absence de fuite de données, tests statistiques) plutôt que sur la recherche d'un
score artificiellement élevé.

## Stack technique
Python, pandas, scikit-learn, LightGBM, CatBoost, XGBoost, scipy (tests statistiques)
