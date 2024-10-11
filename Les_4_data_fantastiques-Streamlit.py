#!/usr/bin/env python3

# Import des librairies

#from IPython.display import display

import streamlit as st
import io
#import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Création d'un dataframe pour lire le data set
df_bank = pd.read_csv("bank.csv", sep = ",")

st.title("Bank Marketing")
st.sidebar.title("Sommaire")
pages=["Le projet","Le jeu de données","Quelques visualisations","Modélisation","Machine Learning","Conclusion"]
page=st.sidebar.radio("Aller vers :", pages)    

# Exploitation d'un fichier .md pour la page de présentation du projet
f = open('contexte_et_objectifs.md')
txt_projet = f.read()

# Zones de texte de la page de présentation du jeu de données
txt_cadre = """
#### Cadre

Le jeu de données dont nous disposons est accessible librement sur la plateforme Kaggle : https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data

Ces données sont sous License CC0: Public Domain
    
La volumétrie du jeu de données est de 17 colonnes pour 11 162 lignes.
    
Les données sont de différentes natures (7 entiers, 6 chaînes de caractères, 4 binaires), et se décomposent en 6 variables quantitatives (age, balance, duration, campaign, pdays, previous) et 11 variables catégorielles (job, marital, education, default, housing, loan, contact, day, month, poutcome, deposit).
"""

txt_pertinence = """
#### Pertinence

La variable la plus importante est la variable binaire « deposit » qui indique si le prospect a finalement souscrit ou non un dépôt à terme suite à la campagne marketing.

Nous devons donc mettre en place un modèle d’apprentissage supervisé suivant la technique de la classification (prédiction d’une variable cible de type qualitatif).

Il nous appartiendra de déterminer la corrélation entre cette variable « deposit » et les autres variables du dataset, pour pouvoir la prédire au final.
"""

txt_preprocess = """
#### Pre-processing et feature engineering
"""

df_pertinence = pd.read_csv("tableau_variables.csv", sep = ";", index_col=0, lineterminator="\n")


txt_conclusion_prepocess = """
#### Conclusions sur les variables du dataset

Nous disposons d’ores et déjà de certaines informations nous permettant de cibler nos futurs prospects. Il semblerait que notre prospect idéal soit :
* âgé de moins de 29 ans ou plus de 60 ans, 
* célibataire, 
* étudiant ou retraité 
* issu un cursus d’études du domaine tertiaire
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
* age : créer des catégories pour faciliter le traitement
* job : supprimer les lignes dont la variable job est manquante
* education : déduire les données manquantes du job associé (modalité education la plus fréquente pour le job en question)
* contact : supprimer la colonne
"""



if page == pages[0]:
    st.header(pages[0])
    st.markdown(txt_projet)
    
if page == pages[1]:
    st.header(pages[1])
    st.markdown(txt_cadre)
    if st.checkbox("Afficher un aperçu des premières lignes du dataframe"):
        st.markdown("**Apercu du dataframe issu du dataset bank :**")
        st.dataframe(df_bank.head())
    if st.checkbox("Afficher un aperçu des données du dataframe"):
        st.markdown("**Apercu de la structure du dataframe :**")
        buffer = io.StringIO()
        df_bank.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.markdown("Les types de variables associés à chaque colonne sont en accord avec ce que représentent ces colonnes.")
    st.markdown(txt_pertinence)
    st.markdown(txt_preprocess)
    st.write("**Nombre de valeurs en doublon dans le dataframe :**", df_bank.duplicated().sum())
#    st.write("**Nombre de valeurs manquantes par colonne :**", df_bank.isna().sum())
    st.markdown("**Valeurs prises par les différentes variables catégorielles :**")
    if st.checkbox("Afficher les valeurs prises par les différentes variables catégorielles"):
        for column in df_bank.select_dtypes(include='object').columns:
            st.markdown("- " + column + ":")
            st.markdown(df_bank[column].unique())
    st.markdown("Le dataframe ne contient aucune valeur manquante à proprement parler mais comprends des champs renseignés unknown à retraiter.")
    st.markdown("**Description des variables quantitatives :**")
    st.table(df_bank.describe())
    if st.checkbox("Afficher le détail des informations sur les variables du dataframe"):
        st.markdown("**Récapitulatif des informations dont nous disposons sur les différentes variables du dataset :**")
        st.table(df_pertinence)
    st.markdown(txt_conclusion_prepocess)





var_num = ["age","balance","duration","campaign","pdays","previous"]
var_cat = ["job","marital","education","default","housing","loan","contact","day","month","poutcome","deposit"]


if page == pages[2]:
    st.header(pages[2])
    st.markdown("#### Répartition des modalités pour chacune des colonnes :")
    choix_type_var = st.selectbox("Choisissez le type de variable à afficher :", ("Variables quantitatives", "Variables catégorielles"))
    if choix_type_var == "Variables quantitatives":
        choix_var_num = st.selectbox("Choisissez la variable quantitative à afficher :", (var_num))
        st.write("##### Distribution de la variable", choix_var_num)
#        fig_num = go.Figure()
#        fig_num.add_trace(go.Box(
#            x=df_bank[choix_var_num],
#            name=choix_var_num,
#            marker_color="#222A2A",
#            opacity=0.7
#            ))
#        fig_num.update_layout(
#            title=("Distribution de la variable", choix_var_num),
#            plot_bgcolor="rgba(0,0,0,0)",
#            xaxis=dict(showgrid=False),
#            yaxis=dict(gridcolor="rgba(210,210,210,0.5)")
#            )
#        st.plotly_chart(fig_num)
        if choix_var_num == "age":
            st.write("Distribution cohérente, concentrée autour de la médiane, pas de valeurs aberrantes.")
            st.write("Nous ne créerons pas de catégories d’âge pour le modèle de Machine Learning. Nous pourrons néanmoins les prévoir pour établir un profil type de clients susceptibles de souscrire un dépôt à terme.")
            st.write("Nous pouvons d’ores et déjà constater que le ratio souscriptions/non-souscriptions est en faveur des prospects âgés de moins de 29 ans ou plus de 60 ans.")
        if choix_var_num == "balance":
            st.write("Distribution cohérente, concentrée autour de la médiane, pas de valeurs aberrantes.")
            st.write("Nous ne créerons pas de catégories pour le modèle de Machine Learning. Nous pourrons néanmoins les prévoir pour étudier un profil type de clients susceptibles de souscrire un dépôt à terme.")
            st.write("Le ratio souscriptions/non-souscriptions est négatif pour les clients dont le solde bancaire est négatif ou faible (inférieur à 800 euros), ce qui parait plutôt cohérent.")
        if choix_var_num == "duration":
            st.markdown("En théorie, pour le Machine Learning, une variable doit être connue a priori, ce qui n’est pas le cas de la variable distribution. Nous verrons par la suite, lors de l’interprétation de nos modèles, si cette variable est importante ou non pour la prédiction.")
            st.markdown("- Si elle figure dans le top 5 des variables utilisées par le modèle, nous la conserverons.") 
            st.markdown("- Sinon, nous la supprimerons pour le Machine Learning, mais nous la conserverons pour émettre des recommandations métier. Cette information peut être utilisée pour valoriser l’intérêt suscité chez le client lors de la campagne passée et l’exploiter pour des prospections futures (plus l’appel était long, plus le client semblait intéressé et donc potentiel prospect pour la prochaine campagne)")
        if choix_var_num == "campaign":
            st.write("Cette information peut être utilisée en complément d’un travail de profiling, pour adapter la méthode d’approche. En l’occurrence, le ratio gain (souscription) / perte (effort fourni pour le démarchage) semble ne plus être intéressant au-delà d’un appel.")
        if choix_var_num == "pdays":
            st.write("Il y a beaucoup de lignes dont pdays est égal à -1. Pour tous les pdays à -1, poutcome est « unknown » (8 324 valeurs). Pour poutcome à « unknown » (8 863 valeurs au total), pdays varie entre -1 et 391.")
            st.write("Nous posons le postulat de départ suivant : pdays à -1 et donc par déduction poutcome à « unknown » indique que le client n’a jamais été contacté auparavant pour une campagne précédente. Il s’agit de nouveaux prospects.")
        if choix_var_num == "previous":
            st.write("Une grande partie des souscriptions lors de cette campagne ont été réalisées par des nouveaux prospects. Néanmoins, le ratio souscription/non-souscription est plus intéressant pour les clients ayant déjà été contactés lors de précédentes campagnes marketing.")
    if choix_type_var == "Variables catégorielles":
        choix_var_cat = st.selectbox("Choisissez la variable catégorielle à afficher :", (var_cat))
        st.write("##### Distribution de la variable", choix_var_cat)
        if choix_var_cat == "job":
            st.write("La variable job comprend très peu de valeurs « unknown » (70 lignes pour une volumétrie totale de plus de 11 000 lignes). Etant donné le faible impact de ces lignes, nous pouvons simplement les supprimer.")
            st.write("Les retraités et étudiants semblent le plus sensibles à la question du dépôt à terme. Cela conforte notre analyse basée sur l’âge (moins de 29 ans et plus de 60 ans).")
        if choix_var_cat == "education":
            st.write("Les clients issus d’études tertiaires semblent plus intéressés par la souscription d’un dépôt à terme.")
            st.write("Les variables job et educations semblent corrélées entre elles. Pour chaque job, une donnée education semble ressortir clairement de manière générale.")
            st.write("Le résultat est plus mitigé pour les retraités et entrepreneurs, mais le nombre de clients issus de ces catégories semble suffisamment faible pour pouvoir en faire abstraction.")
            st.write("Nous pouvons donc en conclure que la variable education, lorsqu’elle est manquante dans le dataset, peut-être déduite de la variable job.")
        if choix_var_cat == "marital":
            st.write("Les clients mariés semblent moins enclins à souscrire un dépôt à terme suite à une campagne marketing. Les célibataires seront de meilleures cibles.")
        if choix_var_cat == "default":
            st.write("Il y a un fort déséquilibre dans la répartition des données : très peu de clients ayant un crédit en défaut sont présents dans le dataset.")
        if choix_var_cat == "housing":
            st.write("Ne pas avoir de crédit immobilier en cours semble favoriser la souscription d’un dépôt à terme.")
        if choix_var_cat == "loan":
            st.write("Il a peu de clients ayant un crédit personnel en cours dans le dataset.")
            st.write("Néanmoins il semblerait qu’un client ayant un crédit personnel en cours soit moins enclin à souscrire un dépôt à terme.")
        if choix_var_cat == "contact":
            st.write("Proportion largement majoritaire de téléphones mobiles, la catégorie « unknown » emporte beaucoup de non-souscriptions. Dans une société où le téléphone portable prime largement sur la ligne fixe et où un certain nombre de ménages ne disposent même pas d’un téléphone fixe et fonctionnent uniquement par téléphone portable, on peut se poser la question de la pertinence de cette variable. Nous faisons le choix de la supprimer du dataset.")
        if choix_var_cat == "day":
            st.write("Certains jours du mois semblent plus favorables au démarchage : les 4 premiers jours du mois, le 10 et le 30 de chaque mois.")
        if choix_var_cat == "month":
            st.write("On peut d’ores et déjà constater que les mois les plus propices au démarchage sont mars, avril, septembre, octobre, décembre. Au contraire, les démarchages ayant lieu en mai donnent de très mauvais retours. Ce mois est celui qui fait l'objet du plus grand nombre de prospections pendant cette campagne, tout en ayant le taux de conversion le plus faible de l'année.")
        if choix_var_cat == "poutcome":
            st.write("Il y a beaucoup de clients pour lesquels nous ne connaissons pas l’issue de la campagne précédente. D’après notre analyse de la variable pdays combinée à la variable poutcome, poutcome à « unknown » nous donne une information à part entière : il s’agit là de nouveaux prospects, jamais contactés auparavant. Nous décidons donc de conserver les 4 catégories de valeurs telles qu’elles sont dans la base de données initiale.")
        if choix_var_cat == "deposit":
            st.write("La variable deposit est notre variable cible. Sa distribution est très équilibrée, et elle ne comprend aucune valeur manquante. Elle ne nécessite donc aucun retraitement.")
    


#       for var in var_cat:
#           plt.title(f"Distribution de la variable {var}")
#           sns.countplot(x = var, data = df_bank)
#           if var == "job":
#               plt.xticks(rotation = 45, ha = "right")
#           plt.show()
    
#print("La variable age indique que les personnes mentionnées dans le dataset ont entre 18 et 95 ans, avec une concentration entre 32 et 49 ans, ce qui est cohérent. Il n'y a pas de valeurs âge aberrante.")
#print("La variable job indique une proportion plus importante d'actifs et une proportion faible d'étudiants et chômeurs.")
#print("La variable marital indique une proportion plus importante de personnes mariées. Viennent ensuite les célibataires, puis les personnes divorcées.")



if page == pages[3]:
    st.header(pages[3])

if page == pages[4]:
    st.header(pages[4])



if page == pages[5]:
    st.header(pages[5])
