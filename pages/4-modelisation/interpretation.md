#### Interprétation des résultats

La variable duration est restée au centre des débats durant tout le projet :
* Elle prend une place prépondérante dans l’ensemble des analyses.
* Toutefois elle n’est connue qu’à posteriori, ce qui interroge sur son utilisabilité en amont.

En effet, pour les modèles pouvant être analysés grâce à leur attribut feature_importances_ ou coef_, la variable duration se trouve dans le top 5 de 6 d’entre-eux. Son importance dans la qualité des prédictions de machine learning n’est donc pas discutable ; les résultats sont globalement supérieurs de 10 points lorsqu’elle est intégrée aux calculs. Toutefois, le temps d’appel, à savoir son essence même, ne peut effectivement pas être connu avant d’avoir passé cet appel.

Devant ce dilemme, nous avons pris la décision d’articuler notre analyse autour de ces deux possibilités, pour ne pas décider arbitrairement d’occulter une face importante de notre étude. Ainsi, tous nos modèles de machine learning sont disponibles en conservant ou non la variable duration. Cela nous permet d'optimiser ou non les scores du machine learning selon notre objectif du moment, tout en conservant des données les plus réalistes possibles selon les besoins. 
Toutefois, pour obtenir un résultat optimal lors de l’application de nos modèles, le métier devra rester conscient du fait que le temps d’appel est une caractéristique à prendre en compte pour améliorer l’efficience du process.

La mise à l'échelle par RobustScaling agit positivement sur les modèles KNN et SVM. Elle est cependant sans incidence significative pour les autres modèles.

Le fait d’encoder la variable education en Ordinal Encoding ou OneHotEncoding semble peu importer. Les résultats sont équivalents avec les 2 méthodes.

L’utilisation d’hyperparamètres ne semble pas avoir d’impact sur nos modèles.

Nous pouvons conclure de tous nos essais que les modèles de base sont très bien optimisés pour l'utilisation dans notre dataset.

Le modèle plus performant est celui du **CatBoosting** utilisé en suivant les paramètres ci-après : 
* avec duration
* RobustScaling sur les variables numériques
* OneHotEncoding pour l'encodage de la variable duration
* sans optimisation des hyperparamètres

C’est donc ce modèle CatBoost qui servira à la conception de notre outil de prédiction. La durée de l’appel représente une importance non négligeable dans le modèle, mais étant donné sa potentielle indisponibilité, il nous parait utile de prévoir deux options. Ainsi, notre outil proposera de renseigner cette donnée en vue de la prédiction, mais permettra également d’obtenir un résultat dans le cas où elle serait absente.
