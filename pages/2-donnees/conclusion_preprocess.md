#### Conclusions sur les variables du dataset

Nous disposons d’ores et déjà de certaines informations nous permettant de cibler nos futurs prospects. Il semblerait que notre prospect idéal soit :
* âgé de moins de 29 ans ou plus de 60 ans,
* célibataire,
* étudiant ou retraité
* issu d'un cursus d’études supérieures
* ayant quelques économies personnelles
* n’ayant aucun crédit en cours
* ayant déjà été contacté lors d’une campagne précédente mais ayant eu lieu il y a moins de 200 jours
* ayant souscrit un produit lors de la campagne précédente.

Nos analyses macro font également ressortir quelques recommandations métier. Les méthodes d’approche à privilégier sont les suivantes :
* contact par appel téléphonique sur une ligne mobile
* avant le 5 du mois ou alors le 10 ou le 30 du mois
* en mars, avril, septembre, octobre ou décembre
* un seul appel par prospect

Pour affiner nos analyses et réussir à prédire si un prospect serait susceptible de souscrire ou non à un dépôt à terme, nous devons passer à l’étape de la modélisation. Néanmoins, avant cela, nous devons procéder aux retraitements suivants :
* job : 1% de valeurs "unknown"(soit 70 lignes), nous avons pris la décision de supprimer ces lignes qui sont peu nombreuses
* education : 4% de valeurs "unknown" (soit 497 lignes), cette variable est importante et ces lignes sont trop nombreuses pour être supprimées. Nous avons donc décidé de compléter les valeurs manquantes. La variable education étant fortement corrélée avec la variable job, nous déduirons les données manquantes du job associé (modalité education la plus fréquente pour le job en question)
* contact : 21% de valeurs "unknown". Après analyse nous avons décidé de supprimer cette colonne qui ne représente pas d'intérêt majeur pour la modélisation.

Pour rappel : la variable poutcome restera inchangée.
En effet, cette variable comporte trop de valeurs manquantes pour pouvoir supprimer des lignes (75% de valeurs "unknown").
De plus, nous avons acté qu'elle était importante pour notre modélisation.
La valeur "unknown" de cette variable constituera donc une valeur à part entière dans notre projet.
