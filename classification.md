#### Classification du problème

Comme nous l’avons vu précédemment, la variable la plus importante de notre jeu de données est la variable binaire « deposit » qui indique si le prospect a finalement souscrit ou non un dépôt à terme suite à la campagne marketing. Après les différents retraitements opérés lors de la phase de pré-processing, nous disposons de 15 variables explicatives et d’1 variable cible de nature catégorielle.

Nous devons donc mettre en place un modèle d’apprentissage supervisé suivant la technique de la classification (prédiction d’une variable cible de type qualitatif).

Le dataset ne présentant pas un grand volume, nous ne réduirons pas sa dimension lors de la modélisation.

Nous diviserons les données en deux parties : 70% du dataset seront dédiés à l’entraînement et 30% du dataset seront dédiés à l’évaluation de notre modèle.

Nos données ne comportent plus de valeurs manquantes mais contiennent encore des données quantitatives extrêmes. Nous avons décidé de les conserver car ces informations restent intéressantes pour notre modèle. Nous ne pouvons pas normaliser ou standardiser les variables concernées car ces techniques sont sensibles aux valeurs extrêmes. Nous devrons en revanche tester s’il est pertinent de les mettre à l’échelle par la technique de Robust Scaling. Les variables concernées sont : age, balance, duration, campaign, pdays, previous.

Les variables catégorielles, quant à elles, devront être encodées de la manière suivante :
* job : OneHotEncoding
* marital : OneHotEncoding
* education : Ordinal Encoding
* default : OneHotEncoding
* housing : OneHotEncoding
* loan : OneHotEncoding
* day : encodage circulaire ou OneHotEncoding ?
* month : conversion texte en numérique ou OneHotEncoding ?
* poutcome : OneHotEncoding
* deposit : LabelEncoding

Métriques de performance principales utilisées pour comparer nos modèles :
* **Accuracy :** L'accuracy est la métrique la plus connue pour évaluer un modèle de classification. Elle correspond simplement au taux de prédictions correctes effectuées par le modèle. Pour rappel, c'est la métrique qui est utilisée par défaut lorsque l'on utilise la méthode score.
* **Précision :** La précision est une métrique qui répond à la question : Parmi toutes les prédictions positives du modèle, combien sont de vrais positifs ? Un score de précision élevé nous informe que le modèle ne classe pas aveuglément toutes les observations comme positives.
* **Rappel (recall en anglais) :** Le rappel est une métrique qui quantifie la proportion d'observations réellement positives qui ont été correctement classifiées positives par le modèle. Un score de rappel élevé nous informe que le modèle est capable de bien détecter les observations réellement positives.
* **Le score F1, ou F1-score en anglais :** Le f1-score est une métrique qui permet de combiner la précision et le rappel, puisqu'elle correspond à leur moyenne harmonique. Le f1-score est une des métriques à privilégier lorsqu'il y a un déséquilibre de classes. En regardant uniquement l'accuracy, les résultats pourraient être faussés.
