#### Classification du problème

Comme nous l’avons vu précédemment, la variable la plus importante de notre jeu de données est la variable binaire « deposit » qui indique si le prospect a finalement souscrit ou non un dépôt à terme suite à la campagne marketing. Après les différents retraitements opérés lors de la phase de pré-processing, nous disposons de 15 variables explicatives et d’1 variable cible de nature catégorielle.

Nous devons donc mettre en place un modèle d’apprentissage supervisé suivant la technique de la classification (prédiction d’une variable cible de type qualitatif).

Nous avons choisi de nettoyer le dataset avant la modélisation car il y avait peu de données à retraiter et que ces retraitements auront peu d'impact sur la suite de notre projet :
* suppression d'une colonne (contact),
* suppression de seulement 70 lignes pour valeur job manquante,
* et remplacement des 427 valeurs manquantes restantes d'education (70 des valeurs manquantes ayant été supprimées avec la suppression des lignes dont job était manquant).

Le dataset ne présentant pas un grand volume, nous ne réduirons pas sa dimension lors de la modélisation.

Nous diviserons les données en deux parties : étant donné que nous disposons d'un dataset d'un volume correct (plus de 10 000 entrées), 70% du dataset seront dédiés à l’entraînement et 30% du dataset seront dédiés à l’évaluation de notre modèle.

**Choix de la métrique de performance**

Selon la classification de notre projet (modèle supervisé de classification), nous pouvons retenir 4 principales métriques de performance : l’exactitude, la précision, le rappel et le score F1.

* **L'exactitude (accuracy en anglais)** évalue le taux de bonnes prédictions par rapport au nombre total de prédictions. Elle est facile à calculer, facile à interpréter, et résume la performance du modèle avec une valeur unique. Elle évalue mal les performances d’un modèle basé sur des données déséquilibrées, mais ce n'est pas le cas dans notre projet. Cependant, l’exactitude ne permet pas de faire des nuances entre les différentes prédictions et néglige de ce fait le coût des faux négatifs. Notre choix se portera donc sur une métrique plus précise.
* **La précision** est une métrique qui répond à la question : Parmi toutes les prédictions positives du modèle, combien sont de vrais positifs ? Un score de précision élevé nous informe que le modèle ne classe pas aveuglément toutes les observations comme positives. Cette métrique est utile lorsque le coût des faux positifs est élevé.
* **Le rappel (recall en anglais)** est une métrique qui quantifie la proportion d'observations réellement positives qui ont été correctement classifiées positives par le modèle (vrai positifs par rapport à la somme des vrais positifs et faux négatifs). Un score de rappel élevé nous informe que le modèle est capable de bien détecter les observations réellement positives. Cette métrique est utile lorsque le coût des faux négatifs est élevé.
* **Le score F1 (F1-score en anglais)** est une métrique qui permet de combiner la précision et le rappel, puisqu'elle correspond à leur moyenne harmonique.

C'est cette dernière métrique, **le score F1**, qui nous parait la plus adaptée pour évaluer nos différents modèles de Machine Learning.

#### Choix du modèle et optimisation

**Choix de l'encodage**

Notre jeu de données ne comporte plus de valeurs manquantes mais contient encore des données quantitatives extrêmes. Nous avons décidé de les conserver car ces informations ne sont pas aberrantes et restent intéressantes pour notre modèle. Nous ne pouvons pas normaliser ou standardiser les variables concernées car ces techniques sont sensibles aux valeurs extrêmes. Nous devrons en revanche tester s’il est pertinent de les mettre à l’échelle par la technique de Robust Scaling. Les variables concernées sont : age, balance, duration, campaign, pdays, previous.

Les variables catégorielles, quant à elles, devront être encodées de la manière suivante :
* job : OneHotEncoding
* marital : OneHotEncoding
* education : Ordinal Encoding ou OneHotEncoding
* default : OneHotEncoding
* housing : OneHotEncoding
* loan : OneHotEncoding
* day : OneHotEncoding
* month : OneHotEncoding
* poutcome : OneHotEncoding
* deposit : LabelEncoding

**Choix des modèles**

Les modèles que nous avons testé sont les suivants :
* Régression logistique (LogiticRegression)
* Machines à Vecteurs de Support (SVC)
* Méthode des K plus proches voisins (KNN - KNeighborsClassifier)
* Arbres de décision (DecisionTreeClassifier et DecisionTreeRegressor)
* Forêts aléatoires (RandomForestClassifier)
* Gradient Boosting (GradientBoostingClassifier)
* Extreme Gradient Boosting (XGBClassifier)
* CatBoosting (CatBoostClassifier)

**Tests réalisés**

Nous avons testé nos modèles avec différents paramètres :
* avec / sans la variable duration
* avec / sans mise à l'échelle des variables numériques (Robust Scaling)
* avec Ordinal Encoding / OneHotEncoding pour la variable education
* avec / sans optimisation des hyperparamètres

**Interprétabilité des modèles**

Nous avons tenté de faire ressortir les variables les plus importantes utilisées par nos différents modèles pour décider du sort de la variable duration. Cependant, nous avons été confrontés à une limite : Les modèles KNN et SVM ne disposent pas de l'attribut feature_importances_ ou coef_. Etant donné qu’il ne s’agit pas des modèles les plus performants et que la grande majorité des modèles dispose de ces attributs, nous avons basé nos analyses sur les résultats disponibles en faisant abstraction des éléments manquants.

**Optimisation des hyperparamètres**

Pour optimiser les hyperparamètres, nous avons fait appel à la méthode GridSearch : celle-ci explore toutes les combinaisons possibles d'hyperparamètres spécifiés pour trouver les meilleurs réglages du modèle.

Nous avons également testé RandomizedSearch et BayerSearch, mais sans résultats probants, bien au contraire. Malgré des tests très longs sur ces hyperparamètres extrêmement lourds à faire tourner même en local, les résultats se trouvaient être moins bons qu'avec GridSearch. Nous avons donc décidé de ne pas intégrer ces tests à notre rapport.

Nous avons testé l'optimisation des hyperparamètres sur la quasi-totalité des modèles sélectionnés. Ces tests ont été réalisés en local car ils sont relativement lourds selon les modèles. Pour éviter de surcharger la plateforme Streamlit et ainsi perdre en efficacité, nous ne présenterons l'optimisation des paramètres que pour les 3 modèles qui nous semblent être les plus performants, à savoir : CatBoosting, Extreme Gradient Boosting et Forêts aléatoires.

**Optimisation du chargement des modèles entraînés**

Enfin, dans le but d'optimiser encore davantage la rapidité de notre plateforme Steamlit, nous avons utilisé le module Pickle. Il s’agit d’une bibliothèque standard de Python qui permet de sérialiser et désérialiser des objets. Elle permet de convertir des objets Python en un format binaire qui peut être enregistré sur le disque et récupéré ultérieurement, ce qui en fait un outil extrêmement utile pour la sauvegarde et la restauration de modèles de machine learning. 

Concrètement, nous avons entraîné en local chacun de nos modèles, l’un après l’autre, en modifiant les paramètres selon nos différents angles d’étude. Pour chaque test réalisé, le module pickle nous a permis de sauvegarder le modèle entraîné ainsi que les meilleurs hyperparamètres utilisés le cas échéant. Seuls ces fichiers sont chargés dans Streamlit, les différents modèles n’ont pas à être réentrainés pour visualiser les résultats de nos tests. Notre plateforme est ainsi plus réactive et plus efficace dans la livraison des résultats des tests.
