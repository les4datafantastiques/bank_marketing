#### Classification du problème

Comme nous l’avons vu précédemment, la variable la plus importante de notre jeu de données est la variable binaire « deposit » qui indique si le prospect a finalement souscrit ou non un dépôt à terme suite à la campagne marketing. Après les différents retraitements opérés lors de la phase de pré-processing, nous disposons de 15 variables explicatives et d’1 variable cible de nature catégorielle.

Nous devons donc mettre en place un modèle d’apprentissage supervisé suivant la technique de la classification (prédiction d’une variable cible de type qualitatif).

Le dataset ne présentant pas un grand volume, nous ne réduirons pas sa dimension lors de la modélisation.

Nous diviserons les données en deux parties : 75% du dataset seront dédiés à l’entraînement et 25% du dataset seront dédiés à l’évaluation de notre modèle.

**Choix des métriques de performance**

Les métriques de performance principales utilisées pour comparer nos modèles sont les suivantes :
* **Accuracy :** L'accuracy est la métrique la plus connue pour évaluer un modèle de classification. Elle correspond simplement au taux de prédictions correctes effectuées par le modèle. Pour rappel, c'est la métrique qui est utilisée par défaut lorsque l'on utilise la méthode score.
* **Précision :** La précision est une métrique qui répond à la question : Parmi toutes les prédictions positives du modèle, combien sont de vrais positifs ? Un score de précision élevé nous informe que le modèle ne classe pas aveuglément toutes les observations comme positives.
* **Rappel (recall en anglais) :** Le rappel est une métrique qui quantifie la proportion d'observations réellement positives qui ont été correctement classifiées positives par le modèle. Un score de rappel élevé nous informe que le modèle est capable de bien détecter les observations réellement positives.
* **Le score F1, ou F1-score en anglais :** Le f1-score est une métrique qui permet de combiner la précision et le rappel, puisqu'elle correspond à leur moyenne harmonique. Le f1-score est une des métriques à privilégier lorsqu'il y a un déséquilibre de classes. En regardant uniquement l'accuracy, les résultats pourraient être faussés.


#### Choix du modèle et optimisation

**Choix de l'encodage**

Notre jeu de données ne comporte plus de valeurs manquantes mais contient encore des données quantitatives extrêmes. Nous avons décidé de les conserver car ces informations restent intéressantes pour notre modèle. Nous ne pouvons pas normaliser ou standardiser les variables concernées car ces techniques sont sensibles aux valeurs extrêmes. Nous devrons en revanche tester s’il est pertinent de les mettre à l’échelle par la technique de Robust Scaling. Les variables concernées sont : age, balance, duration, campaign, pdays, previous.

Les variables catégorielles, quant à elles, devront être encodées de la manière suivante :
* job : OneHotEncoding
* marital : OneHotEncoding
* education : Ordinal Encoding ou OneHotEncoding
* default : OneHotEncoding
* housing : OneHotEncoding
* loan : OneHotEncoding
* day : OneHotEncoding
* month : OneHotEncoding
* poutcome : OneHotEncoding
* deposit : LabelEncoding

**Choix des modèles**

Les modèles que nous avons testé sont les suivants :
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

**Optimisation des hyperparamètres**

Pour optimiser les hyperparamètres, nous avons fait appel à la méthode GridSearch : celle-ci explore toutes les combinaisons possibles d'hyperparamètres spécifiés pour trouver les meilleurs réglages du modèle.

Nous avons testé l'optimisation des hyperparamètres sur la quasi-totalité des modèles sélectionnés. Ces tests ont été réalisés en local car ils sont relativement lourds selon les modèles. Pour éviter de surcharger la plateforme Streamlit et ainsi perdre en efficacité, nous ne présenterons l'optimisation des paramètres que pour les 3 modèles qui nous semblent être les plus performants, à savoir : ..., ... et ... .


#### Interprétation des résultats

Lors de l’exploration des données, nous nous posions la question de la conservation ou non de la variable « duration ».
Nous n'avons pas pu faire resortir les variables les plus importantes pour tous nos modèles car certains ne possèdent pas d'attribut feature_importances_ ou coef_. Néanmoins, la variable duration semble être dans le top 5 des variables les plus importantes pour au moins 4 de nos modèles. Nous pourrions donc en déduire qu'elle joue un rôle prépondérant dans notre projet de prédiction et que nous ne pouvons donc pas la supprimer de notre base. Néanmoins, nous avons décidé d'interpréter ce résultat différemment : cette variable n'étant pas connue a priori, la place qu'elle semble prendre dans la prédiction est trop importante. Nous prenons donc le parti de la supprimer pour notre modèle de prédiction.

La mise à l'échelle par RobustScaling agit positivement sur les modèles suivants : Logistic Regression, SVM, KNN, Decision Tree Classifier, Decision Tree Regressor et Random Forest.
Elle est cependant sans incidence pour les modèles Gradient Boost, Extreme Gradient Boost et CatBoost.

Le fait d’encoder la variable education en Ordinal Encoding ou OneHotEncoding semble peu importer. Les résultats sont équivalents avec les 2 méthodes.
