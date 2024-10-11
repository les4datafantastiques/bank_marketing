#### Contexte

##### Contexte d’insertion du projet dans votre métier ou intérêts professionnels

Le projet « bank marketing » nous présente en tant que data analystes pour une banque. La direction de la société nous demande de trouver les meilleures stratégies pour améliorer les prochaines campagnes marketing. Sur la base des informations collectées lors de la dernière campagne, nous devrons identifier les modèles les plus efficaces pour atteindre un objectif bien précis, le succès d’une future campagne de souscription.

Ce projet permet de conjuguer l’analyse avancée de données avec une compréhension approfondie d’enjeux économiques et financiers pour la banque. L’idée est d’améliorer les processus et de fournir à la direction des recommandations stratégiques visant à favoriser la réussite globale de l'entreprise.

##### Dimension Technique

**Utilisation de Python de A à Z :** Nous utiliserons Python au maximum des concepts que nous avons appris pendant notre formation, avec des bibliothèques telles que Pandas pour le traitement des données, Seaborn et Matplotlib pour la visualisation des résultats, et Scikit-learn pour l’analyse prédictive.

Nous articulerons notre progression autour de quatre thèmes principaux.

1. Préparation et nettoyage des données :
    * Traitement des valeurs manquantes : identifier et corriger les données manquantes pour éviter les biais dans le modèle (utilisation de médianes, moyennes, imputation des k proches…)
    * Encodage des variables catégorielles : convertir les variables qualitatives en formats numériques adaptés aux algorithmes de Machine Learning (one hot, label encoding, ordinal encoding)
    * Ajustement des variables : normaliser ou standardiser les variables pour assurer une cohérence dans la modélisation.
2. Analyse exploratoire des données :
    * Identification des schémas : analyser les distributions des données pour détecter des motifs récurrents et des tendances significatives.
    * Exploration des relations : utiliser des visualisations et des statistiques descriptives pour examiner les corrélations entre les variables explicatives et la variable cible (histogrammes, boxplots, heatmaps…).
3. Modélisation prédictive :
    * Sélection et entrainement des modèles : tester et entraîner divers algorithmes de Machine Learning pour déterminer le modèle le plus performant (forêts aléatoires, régressions logistiques, classificateur d’arbre de décision…).
    * Évaluation des performances : mesurer l'efficacité des modèles à l'aide de métriques telles que l’accuracy, la précision, les MAE et MSE, ou encore le f1_score, pour sélectionner le modèle optimal.
4. Développement d'un rapport interactif avec Streamlit :
    * Création du rapport : utiliser Streamlit pour développer un rapport interactif permettant de présenter les résultats de l'analyse et de la modélisation de manière dynamique et accessible. 
    * Présentation des résultats : permettre aux utilisateurs d'explorer les résultats et les insights via une interface web simple, où ils pourront interagir avec les données, filtrer les résultats et visualiser les performances des modèles en temps réel.

##### Dimension Économique

**Amélioration du processus de prise de décision stratégique :** l’utilisation d’un modèle de machine learning permet une analyse facilitée des données dans le but de formuler des recommandations stratégiques pour l’optimisation des résultats des campagnes futures. Cela soutient une prise de décision plus éclairée, permettant de saisir de nouvelles opportunités économiques et limiter les coût financiers d’une campagne marketing.

**Innovation pour la compétitivité :** en introduisant de nouvelles approches analytiques et en optimisant les processus, le projet contribue à l'innovation continue. Cela renforce la compétitivité de l'entreprise qui adopte des pratiques de gestion plus efficaces et améliore la réponse aux évolutions du marché.

#### Objectifs

Nous disposons de données personnelles sur les prospects de la banque ayant été contactés pour souscrire à un "dépôt à terme", ainsi que les résultats de ces opérations (souscription ou non du dépôt à terme suite au télémarketing).

Notre objectif est de prédire le succès d'une campagne marketing en déterminant à l'avance si un client est susceptible de souscrire au produit. Cette prédiction permettra d'optimiser le déploiement de la campagne, en maximisant le taux de souscriptions par rapport au nombre de prospections réalisées.

Pour atteindre cet objectif, nous devrons :
* Cibler les prospects les plus susceptibles de souscrire à un dépôt à terme en établissant un profil type afin d'éviter les prospections inefficaces.
* Déterminer l'approche la plus efficace en optimisant la stratégie de contact pour améliorer les taux de conversion
