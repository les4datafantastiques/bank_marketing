#### Interprétation des résultats

Lors de l’exploration des données, nous nous posions la question de la conservation ou non de la variable « duration ».
Nous n'avons pas pu faire ressortir les variables les plus importantes pour tous nos modèles car certains ne possèdent pas d'attribut feature_importances_ ou coef_. Néanmoins, la variable duration semble être dans le top 5 des variables les plus importantes pour au moins 6 de nos modèles. Nous pourrions donc en déduire qu'elle joue un rôle prépondérant dans notre projet de prédiction et que nous ne pouvons donc pas la supprimer de notre base. Néanmoins, nous avons décidé d'interpréter ce résultat différemment : cette variable n'étant pas connue a priori, la place qu'elle semble prendre dans la prédiction est trop importante. Nous prenons donc le parti de la supprimer pour notre modèle de prédiction.

La mise à l'échelle par RobustScaling agit positivement sur les modèles suivants : Logistic Regression, SVM, KNN, Decision Tree Classifier, Decision Tree Regressor et Random Forest.
Elle est cependant sans incidence pour les modèles Gradient Boost, Extreme Gradient Boost et CatBoost.

Le fait d’encoder la variable education en Ordinal Encoding ou OneHotEncoding semble peu importer. Les résultats sont équivalents avec les 2 méthodes.
