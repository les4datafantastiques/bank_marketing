#!/usr/bin/env python3

# Import des librairies

#from IPython.display import display

import streamlit as st
import io
#import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
#from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
import shap
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

from pathlib import Path

project_root = Path('.')
data_path = project_root / 'data'
build_path = project_root / 'build'
build_df_path = build_path / 'df'
build_graphs_path = build_path / 'graphs'
build_ml_path = build_path / 'ml'
pages_path = project_root / 'pages'
projet_path = pages_path / '1-projet'
donnees_path = pages_path / '2-donnees'
visu_path = pages_path / '3-visu'
modelisation_path = pages_path / '4-modelisation'
ml_path = pages_path / '5-ml'
conclusion_path = pages_path / '6-conclusion'


# Création d'un dataframe pour lire le data set
df_bank = pd.read_csv(data_path / "bank.csv", sep = ",")

st.title("Bank Marketing")
st.sidebar.title("Sommaire")
projet, donnees, visu, modelisation, ml, outil, conclusion = ("Le projet","Le jeu de données","Quelques visualisations","Modélisation","Machine Learning","Outil de prédiction","Conclusion")
pages=[projet, donnees, visu, modelisation, ml, outil, conclusion]
page=st.sidebar.radio("Aller vers :", pages)    


icone_LinkedIn = "https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg"
texte_icone_LinkedIn = "LinkedIn"
taille_icone_LinkedIn = "16"
format_icone_LinkedIn = "position: relative; top: -2px; margin-left: 5px;"

st.sidebar.markdown(
    f"""
    <div style="border: 2px dashed #ff4b4b; padding: 10px; border-radius: 5px; margin-top: 20px;">
        <p style="font-size: 14px; text-align: left; font-weight: bold; padding-left: 20px; margin: 0px;">
            Projet réalisé par :
        </p>
        <ul style="font-size: 14px; list-style-type: disc; padding-left: 20px; margin: 0px;">
	     <li>Arnaud Leleu
            <a href="https://www.linkedin.com/in/arnaud-leleu-1823a9b1/" target="_blank" style="text-decoration: none;">
            <img src="{icone_LinkedIn}" alt="{texte_icone_LinkedIn}" width="{taille_icone_LinkedIn}" style="{format_icone_LinkedIn}">
            </a>
            </li>
         <li>Camille Kienlen
            <a href="https://www.linkedin.com/in/camille-kienlen/" target="_blank" style="text-decoration: none;">
            <img src="{icone_LinkedIn}" alt="{texte_icone_LinkedIn}" width="{taille_icone_LinkedIn}" style="{format_icone_LinkedIn}">
            </a>
            </li>
	     <li>Clément Guillet
            <a href="https://www.linkedin.com/in/cl%C3%A9ment-guillet-975a182b8/" target="_blank" style="text-decoration: none;">
            <img src="{icone_LinkedIn}" alt="{texte_icone_LinkedIn}" width="{taille_icone_LinkedIn}" style="{format_icone_LinkedIn}">
            </a>
            </li>
	     <li>Julien Musschoot
            <a href="https://www.linkedin.com/in/julien-musschoot-0b2b50171/" target="_blank" style="text-decoration: none;">
            <img src="{icone_LinkedIn}" alt="{texte_icone_LinkedIn}" width="{taille_icone_LinkedIn}" style="{format_icone_LinkedIn}">
            </a>
            </li>
         </ul>
    </div>
    """,
    unsafe_allow_html=True
)


# Exploitation d'un fichier .md pour la page de présentation du projet
txt_projet = open(projet_path / 'contexte_et_objectifs.md').read()

# Fichiers exploités sur la page de présentation du jeu de données
txt_cadre = open(donnees_path / 'cadre.md').read()
txt_pertinence = open(donnees_path / 'pertinence.md').read()
df_pertinence = pd.read_csv(donnees_path / "tableau_variables.csv", sep = ";", index_col=0, lineterminator="\n")
txt_conclusion_prepocess = open(donnees_path / 'conclusion_preprocess.md').read()

# Fichiers exploités pour la page de modélisation du projet
txt_classif_choix = open(modelisation_path / 'classification_choix.md').read()
txt_interpretation = open(modelisation_path / 'interpretation.md').read()

# Fichier exploité pour la page de conclusion du projet
txt_conclusion_generale = open(conclusion_path / 'conclusion.md').read()

if page == projet:
    st.header(projet)
    st.markdown(txt_projet)
    
if page == donnees:
    st.header(donnees)
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
    st.markdown("#### Pre-processing et feature engineering")
    st.write("**Nombre de valeurs en doublon dans le dataframe :**", df_bank.duplicated().sum())
#   st.write("**Nombre de valeurs manquantes par colonne :**", df_bank.isna().sum())
    st.markdown("**Valeurs prises par les différentes variables catégorielles :**")
    if st.checkbox("Afficher les valeurs prises par les différentes variables catégorielles"):
        for column in df_bank.select_dtypes(include='object').columns:
            st.markdown("- " + column + ":")
            st.markdown(df_bank[column].unique())
    st.markdown("Le dataframe ne contient aucune valeur manquante à proprement parler mais comprends des champs renseignés unknown à retraiter.")
    st.markdown("**Description des variables quantitatives :**")
    st.table(df_bank.describe().astype(int).map(lambda x: f"{x:,}".replace(",", " ")).style.set_properties(**{'text-align': 'right'}))
    if st.checkbox("Afficher le détail des informations sur les variables du dataframe"):
        st.markdown("**Récapitulatif des informations dont nous disposons sur les différentes variables du dataset :**")
        st.table(df_pertinence)
    st.markdown(txt_conclusion_prepocess)





var_num = ["age","balance","duration","campaign","pdays","previous"]
var_cat = ["job","marital","education","default","housing","loan","contact","day","month","poutcome","deposit"]

jobs_education = {}
for element in df_bank["job"].unique():
    selection_job = df_bank.loc[df_bank["job"]== element]
    valeurs_job = selection_job["education"].value_counts()
    mode_job = selection_job["education"].mode()[0]
    jobs_education[element] = mode_job

if page == visu:
    st.header(visu)
    st.markdown("#### Répartition des modalités pour chacune des colonnes :")
    choix_type_var = st.selectbox("Choisissez le type de variable à afficher :", ("Variables quantitatives", "Variables catégorielles"))
    if choix_type_var == "Variables quantitatives":
        choix_var_num = st.selectbox("Choisissez la variable quantitative à afficher :", (var_num))
#       st.write("##### Variable", choix_var_num)
#       Boxplot de distribution de la variable quantitative
        fig_num = go.Figure()
        fig_num.add_trace(go.Box(
            x=df_bank[choix_var_num],
            name=choix_var_num,
            marker_color="#222A2A",
            opacity=0.7
            ))
        fig_num.update_layout(
            title=("Distribution de la variable " + choix_var_num),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(210,210,210,0.5)")
            )
        st.plotly_chart(fig_num)
        if choix_var_num == "age":
            st.write("Distribution cohérente, concentrée autour de la médiane, pas de valeurs aberrantes.")
#           Histogramme age / deposit
            fig_var_num_deposit = go.Figure()
            fig_var_num_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "no"][choix_var_num],
                name="No",
                marker_color="#222A2A",
                opacity=0.7,
                nbinsx=75
                ))
            fig_var_num_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "yes"][choix_var_num],
                name="Yes",
                marker_color="#19D3F3",
                opacity=0.7,
                nbinsx=75
                ))
            fig_var_num_deposit.update_layout(
                title="Souscription au dépôt selon la variable " + choix_var_num,
                xaxis_title=choix_var_num,
                yaxis_title="Nombre de clients",
                barmode="group",
                legend_title="Souscription",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(210,210,210,0.5)")
                )
            st.plotly_chart(fig_var_num_deposit)
            st.write("Nous pouvons d’ores et déjà constater que le ratio souscriptions/non-souscriptions est en faveur des prospects âgés de moins de 29 ans ou plus de 60 ans.")
        if choix_var_num == "balance":
            st.write("Distribution cohérente, concentrée autour de la médiane, pas de valeurs aberrantes.")
#           Histogramme balance / deposit
            fig_var_num_deposit = go.Figure()
            fig_var_num_deposit.add_trace(go.Histogram(
                x=df_bank[(df_bank["deposit"] == "no") & (df_bank["balance"] > -800) & (df_bank["balance"] < 4000)][choix_var_num],
                name="No",
                marker_color="#222A2A",
                opacity=0.7,
                nbinsx=25
                ))
            fig_var_num_deposit.add_trace(go.Histogram(
                x=df_bank[(df_bank["deposit"] == "yes") & (df_bank["balance"] > -800) & (df_bank["balance"] < 4000)][choix_var_num],
                name="Yes",
                marker_color="#19D3F3",
                opacity=0.7,
                nbinsx=25
                ))
            fig_var_num_deposit.update_layout(
                title="Souscription au dépôt selon la variable " + choix_var_num,
                xaxis_title=choix_var_num,
                yaxis_title="Nombre de clients",
                barmode="group",
                legend_title="Souscription",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(210,210,210,0.5)")
                )
            st.plotly_chart(fig_var_num_deposit)
            st.write("Le ratio souscriptions/non-souscriptions est négatif pour les clients dont le solde bancaire est négatif ou faible (inférieur à 800 euros), ce qui parait plutôt cohérent.")
        if choix_var_num == "duration":
            st.markdown("En théorie, pour le Machine Learning, une variable doit être connue a priori, ce qui n’est pas le cas de la variable distribution. Nous verrons par la suite, lors de l’interprétation de nos modèles, si cette variable est importante ou non pour la prédiction.")
            st.markdown("- Si elle figure dans le top 5 des variables utilisées par le modèle, nous la conserverons.") 
            st.markdown("- Sinon, nous la supprimerons pour le Machine Learning, mais nous la conserverons pour émettre des recommandations métier. Cette information peut être utilisée pour valoriser l’intérêt suscité chez le client lors de la campagne passée et l’exploiter pour des prospections futures (plus l’appel était long, plus le client semblait intéressé et donc potentiel prospect pour la prochaine campagne)")
        if choix_var_num == "campaign":
#           Histogramme campaign / deposit
            fig_var_num_deposit = go.Figure()
            fig_var_num_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "no"][df_bank["campaign"] <= 8][choix_var_num],
                name="No",
                marker_color="#222A2A",
                opacity=0.7,
                nbinsx=8
                ))
            fig_var_num_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "yes"][df_bank["campaign"] <= 8][choix_var_num],
                name="Yes",
                marker_color="#19D3F3",
                opacity=0.7,
                nbinsx=8
                ))
            fig_var_num_deposit.update_layout(
                title="Souscription au dépôt selon le nombre d'appels",
                xaxis_title="Nombre d'appels",
                yaxis_title="Nombre de clients",
                barmode="group",
                legend_title="Souscription",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(210,210,210,0.5)")
                )
            st.plotly_chart(fig_var_num_deposit)
            st.write("Cette information peut être utilisée en complément d’un travail de profiling, pour adapter la méthode d’approche. En l’occurrence, le ratio gain (souscription) / perte (effort fourni pour le démarchage) semble ne plus être intéressant au-delà d’un appel.")
        if choix_var_num == "pdays":
#           Histogramme pdays / deposit
            fig_var_num_deposit = go.Figure()
            fig_var_num_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "no"][choix_var_num],
                name="No",
                marker_color="#222A2A",
                opacity=0.7,
                nbinsx=10
                ))
            fig_var_num_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "yes"][choix_var_num],
                name="Yes",
                marker_color="#19D3F3",
                opacity=0.7,
                nbinsx=10
                ))
            fig_var_num_deposit.update_layout(
                title="Souscription au dépôt selon le nombre de jours depuis la dernière campagne",
                xaxis_title="Nombre de jours depuis la dernière campagne",
                yaxis_title="Nombre de clients",
                barmode="group",
                legend_title="Souscription",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(210,210,210,0.5)")
                )
            st.plotly_chart(fig_var_num_deposit)
            st.write("Il y a beaucoup de lignes dont pdays est égal à -1. Pour tous les pdays à -1, poutcome est « unknown » (8 324 valeurs). Pour poutcome à « unknown » (8 863 valeurs au total), pdays varie entre -1 et 391.")
            st.write("Nous posons le postulat de départ suivant : pdays à -1 et donc par déduction poutcome à « unknown » indique que le client n’a jamais été contacté auparavant pour une campagne précédente. Il s’agit de nouveaux prospects.")
        if choix_var_num == "previous":
#           Histogramme previous / deposit
            fig_var_num_deposit = go.Figure()
            fig_var_num_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "no"][df_bank["previous"]<10][choix_var_num],
                name="No",
                marker_color="#222A2A",
                opacity=0.7,
                nbinsx=10
                ))
            fig_var_num_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "yes"][df_bank["previous"]<10][choix_var_num],
                name="Yes",
                marker_color="#19D3F3",
                opacity=0.7,
                nbinsx=10
                ))
            fig_var_num_deposit.update_layout(
                title="Souscription au dépôt selon le nombre de contacts pré-campagne",
                xaxis_title="Nombre de contacts pré-campagne",
                yaxis_title="Nombre de clients",
                barmode="group",
                legend_title="Souscription",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(210,210,210,0.5)")
                )
            st.plotly_chart(fig_var_num_deposit)
            st.write("Une grande partie des souscriptions lors de cette campagne ont été réalisées par des nouveaux prospects. Néanmoins, le ratio souscription/non-souscription est plus intéressant pour les clients ayant déjà été contactés lors de précédentes campagnes marketing.")
    if choix_type_var == "Variables catégorielles":
        choix_var_cat = st.selectbox("Choisissez la variable catégorielle à afficher :", (var_cat))
#       st.write("##### Variable", choix_var_cat)
#       Histogramme variable catégorielle / deposit
        if choix_var_cat not in ("day", "month"):
            fig_var_cat_deposit = go.Figure()
            fig_var_cat_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "no"][choix_var_cat],
                name="No",
                marker_color="#222A2A",
                opacity=0.7,
                nbinsx=12
                ))
            fig_var_cat_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "yes"][choix_var_cat],
                name="Yes",
                marker_color="#19D3F3",
                opacity=0.7,
                nbinsx=12
                ))
            fig_var_cat_deposit.update_layout(
                title="Souscription au dépôt selon la variable " + choix_var_cat,
                xaxis_title=choix_var_cat,
                yaxis_title="Nombre de clients",
                barmode="group",
                legend_title="Souscription",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(210,210,210,0.5)")
                )
            st.plotly_chart(fig_var_cat_deposit)
        if choix_var_cat == "job":
            st.write("La variable job comprend très peu de valeurs « unknown » (70 lignes pour une volumétrie totale de plus de 11 000 lignes). Etant donné le faible impact de ces lignes, nous pouvons simplement les supprimer.")
            st.write("Les retraités et étudiants semblent le plus sensibles à la question du dépôt à terme. Cela conforte notre analyse basée sur l’âge (moins de 29 ans et plus de 60 ans).")
        if choix_var_cat == "education":
            st.write("Les clients issus d’études tertiaires semblent plus intéressés par la souscription d’un dépôt à terme.")
#           Répartition job par niveau d'éducation
            colors = ["#19D3F3", "#4B4B4B", "#1E90FF", "#060808"]
            category_order = ["primary", "secondary", "tertiary", "unknown"]
            fig_educ_job = go.Figure()
            for i, education in enumerate(category_order):
                if education in df_bank["education"].unique():
                    fig_educ_job.add_trace(go.Histogram(
                    x=df_bank[df_bank["education"] == education]["job"],
                    name=education,
                    marker_color=colors[i],
                    opacity=0.7
                    ))
            fig_educ_job.update_layout(
                title="Distribution de la variable education par job",
                xaxis_title="Job",
                yaxis_title="Nombre de clients",
                barmode="group",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(210,210,210,0.5)"),
                showlegend=True
                )
            st.plotly_chart(fig_educ_job)
            st.write("Les variables job et educations semblent corrélées entre elles. Pour chaque job, une donnée education semble ressortir clairement de manière générale.")
            st.write("Le résultat est plus mitigé pour les retraités et entrepreneurs, mais le nombre de clients issus de ces catégories semble suffisamment faible pour pouvoir en faire abstraction.")
            st.write("Nous pouvons donc en conclure que la variable education, lorsqu’elle est manquante dans le dataset, peut être déduite de la variable job.")
            st.write("Modalités les plus fréquentes de education par job :")
            st.write(jobs_education)
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
#           Histogramme deposit / jour de l'appel
            fig_day_deposit = go.Figure()
            fig_day_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "no"][choix_var_cat],
                name="No",
                marker_color="#222A2A",
                opacity=0.7,
                nbinsx=31
                ))
            fig_day_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "yes"][choix_var_cat],
                name="Yes",
                marker_color="#19D3F3",
                opacity=0.7,
                nbinsx=31
                ))
            fig_day_deposit.update_layout(
                title="Souscription au dépôt selon le jour de l'appel",
                xaxis_title="Jour de l'appel dans le mois",
                yaxis_title="Nombre de clients",
                barmode="group",
                legend_title="Souscription",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(210,210,210,0.5)")
                )
            st.plotly_chart(fig_day_deposit)
            st.write("Certains jours du mois semblent plus favorables au démarchage : les 4 premiers jours du mois, le 10 et le 30 de chaque mois.")
        if choix_var_cat == "month":
#           Histogramme deposit / mois de l'appel
#           Création d'un ordre calendaire pour clarifier le graphique suivant
            month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            df_bank['month'] = pd.Categorical(df_bank['month'], categories=month_order, ordered=True)
            df_bank['month_numeric'] = df_bank['month'].cat.codes
            fig_month_deposit = go.Figure()
            fig_month_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "no"]['month_numeric'],
                name="No",
                marker_color="#222A2A",
                opacity=0.7,
                nbinsx=len(month_order)
                ))
            fig_month_deposit.add_trace(go.Histogram(
                x=df_bank[df_bank["deposit"] == "yes"]['month_numeric'],
                name="Yes",
                marker_color="#19D3F3",
                opacity=0.7,
                nbinsx=len(month_order)
                ))
            fig_month_deposit.update_layout(
                title="Souscription au dépôt selon le mois de l'appel",
                xaxis_title="Mois de l'appel",
                yaxis_title="Nombre de clients",
                barmode="group",
                legend_title="Souscription",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="rgba(210,210,210,0.5)"),
                xaxis=dict(showgrid=False, 
                    tickvals=list(range(len(month_order))),
                    ticktext=month_order
                    )
                )
            st.plotly_chart(fig_month_deposit)
            st.write("On peut d’ores et déjà constater que les mois les plus propices au démarchage sont mars, avril, septembre, octobre, décembre. Au contraire, les démarchages ayant lieu en mai donnent de très mauvais retours. Ce mois est celui qui fait l'objet du plus grand nombre de prospections pendant cette campagne, tout en ayant le taux de conversion le plus faible de l'année.")
        if choix_var_cat == "poutcome":
            st.write("Il y a beaucoup de clients pour lesquels nous ne connaissons pas l’issue de la campagne précédente. D’après notre analyse de la variable pdays combinée à la variable poutcome, poutcome à « unknown » nous donne une information à part entière : il s’agit là de nouveaux prospects, jamais contactés auparavant. Nous décidons donc de conserver les 4 catégories de valeurs telles qu’elles sont dans la base de données initiale.")
        if choix_var_cat == "deposit":
            st.write("La variable deposit est notre variable cible. Sa distribution est très équilibrée, et elle ne comprend aucune valeur manquante. Elle ne nécessite donc aucun retraitement.")



# RETRAITEMENTS POUR GENERER df_bank_0 (Base utilisée uniquement pour le profiling)

# Création d'un dataframe df_bank_0 copie de df_bank :
df_bank_0 = df_bank.copy()
# Création d'une fonction qui définit la catégorie d'âge sur la base de "age":
def get_categ(age):
	if age <= 31:
		categ = "extreme_bas"
	elif 31 < age <= 40:
		categ = "jeune"
	elif 30 < age <= 49:
		categ = "moins_jeune"
	elif 49 < age:
		categ = "extreme_haut"
	return categ
df_bank_0["age_categ"] = df_bank_0["age"].apply(get_categ)
# Export du dataframe df_bank_1 en fichier .csv
#df_bank_0.to_csv(build_df_path / 'bank_0_profiling.csv', index=False, sep=',')


# RETRAITEMENTS POUR GENERER df_bank_1 (retraitements de base)

# Suppression des lignes dont la valeur "job" est manquante :
df_bank_1 = df_bank.loc[df_bank["job"] != "unknown"]
# Remplacement des unknown de "education" par la modalité la plus fréquente rencontrée pour un "job" identique :
df_bank_1.loc[df_bank_1["education"] == "unknown", "education"] = df_bank_1.loc[df_bank_1["education"] == "unknown", "job"].map(jobs_education)		
# Suppression de la colonne "contact" :
df_bank_1 = df_bank_1.drop("contact", axis=1)
# Export du dataframe df_bank_1 en fichier .csv
#df_bank_1.to_csv(build_df_path / 'bank_1.csv', index=False, sep=',')


# RETRAITEMENTS SUPPLEMENTAIRES POUR GENERER df_bank_2 (sans duration)

# Suppression de la colonne "duration" :
df_bank_2 = df_bank_1.drop("duration", axis=1)
# Export du dataframe df_bank_2 en fichier .csv
#df_bank_2.to_csv(build_df_path / 'bank_2.csv', index=False, sep=',')


# Mise en place du tronc commun à tous nos tests de machine learning

le = LabelEncoder()
ohe = OneHotEncoder(drop = "first")
oe = OrdinalEncoder(categories = [["primary", "secondary", "tertiary"]])
num_scaler = RobustScaler()

models = {
        "CatBoost" : CatBoostClassifier(silent = True),  # 'silent=True' pour éviter les logs
        "Random Forest": RandomForestClassifier(),
        "Extreme Gradient Boost" : XGBClassifier(),
        "Gradient Boost" : GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression(max_iter = 1000),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "SVM": SVC(),
        "KNN" : neighbors.KNeighborsClassifier()
        }


if page == modelisation:
    st.header(modelisation)
    st.markdown(txt_classif_choix)
    st.markdown(txt_interpretation)



if page == ml:
    st.header(ml)
    traitement_duration = st.radio("Choisissez le traitement à appliquer à la colonne duration :", ("Conserver la colonne duration", "Supprimer la colonne duration"))
    traitement_var_num = st.radio("Choisissez le traitement des variables numériques :", ("Avec RobustScaling", "Sans RobustScaling"))
    traitement_education = st.radio("Choisissez le traitement de la variable education :", ("Ordinal Encoding", "OneHotEncoding"))
    st.write("Pour des raisons d'optimisation des performances de la plateforme, nous ne ferons tourner GridSearch que sur les 3 modèles qui nous semblent être les plus performants, à savoir : CatBoosting, Extreme Gradient Boosting et Forêts aléatoires.")
    optimisation_hyperparam = st.radio("Souhaitez-vous optimiser les hyperparamètres pour les 3 modèles les plus performants ?", ("Oui", "Non"))
    if traitement_duration == "Conserver la colonne duration":
        df = df_bank_1
        var_num = ["age","balance","duration","campaign","pdays","previous"]
    else :
        df = df_bank_2
        var_num = ["age","balance","campaign","pdays","previous"]
    data = df.drop("deposit", axis = 1)
    target = df["deposit"]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.30, random_state = 88)
    if traitement_var_num == "Avec RobustScaling":
        if traitement_education == "Ordinal Encoding":
            var_cat_for_ohe = ["job","marital","default","housing","loan","day","month","poutcome"]
            prepro = ColumnTransformer(transformers = [("numerical", num_scaler, var_num), ("categorical_ohe", ohe, var_cat_for_ohe), ("categorical_oe", oe, ["education"])])
        else:
            var_cat_for_ohe = ["job","marital", "education","default","housing","loan","day","month","poutcome"]
            prepro = ColumnTransformer(transformers = [("numerical", num_scaler, var_num), ("categorical_ohe", ohe, var_cat_for_ohe)])
    else :
        if traitement_education == "Ordinal Encoding":
            var_cat_for_ohe = ["job","marital","default","housing","loan","day","month","poutcome"]
            prepro = ColumnTransformer(transformers = [("numerical", "passthrough", var_num),("categorical_ohe", ohe, var_cat_for_ohe), ("categorical_oe", oe, ["education"])])
        else : 
            var_cat_for_ohe = ["job","marital", "education","default","housing","loan","day","month","poutcome"]
            prepro = ColumnTransformer(transformers = [("numerical", "passthrough", var_num),("categorical_ohe", ohe, var_cat_for_ohe)])
    X_train_prepro = prepro.fit_transform(X_train)
    X_test_prepro = prepro.transform(X_test)
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    resultats = []
    features = {}
    best_params_list = {}
#    """
#        Attribuer à num_test la valeur du test souhaité
#        Catégorie de test et valeurs possibles de num_test :
#            1 : avec la variable duration / sans Robust Scaling / Ordinal Encoding pour education / sans optimisation des hyperparamètres 
#            2 : avec la variable duration / avec Robust Scaling / Ordinal Encoding pour education / sans optimisation des hyperparamètres
#            3 : sans la variable duration / sans Robust Scaling / Ordinal Encoding pour education / sans optimisation des hyperparamètres
#            4 : sans la variable duration / avec Robust Scaling / Ordinal Encoding pour education / sans optimisation des hyperparamètres
#            5 : sans la variable duration / avec Robust Scaling / OneHotEncoding pour education / sans optimisation des hyperparamètres 
#            6 : sans la variable duration / avec Robust Scaling / Ordinal Encoding pour education / avec optimisation des hyperparamètres 
#            7 : avec la variable duration / sans Robust Scaling / OneHotEncoding pour education / sans optimisation des hyperparamètres 
#            8 : avec la variable duration / avec Robust Scaling / OneHotEncoding pour education / sans optimisation des hyperparamètres
#            9 : avec la variable duration / sans Robust Scaling / Ordinal Encoding pour education / avec optimisation des hyperparamètres 
#            10 : avec la variable duration / avec Robust Scaling / Ordinal Encoding pour education / avec optimisation des hyperparamètres
#            11 : avec la variable duration / sans Robust Scaling / OneHotEncoding pour education / avec optimisation des hyperparamètres 
#            12 : sans la variable duration / avec Robust Scaling / OneHotEncoding pour education / avec optimisation des hyperparamètres
#            13 : sans la variable duration / sans Robust Scaling / Ordinal Encoding pour education / avec optimisation des hyperparamètres
#            14 : sans la variable duration / sans Robust Scaling / OneHotEncoding pour education / sans optimisation des hyperparamètres
#            15 : sans la variable duration / sans Robust Scaling / OneHotEncoding pour education / avec optimisation des hyperparamètres
#            16 : avec la variable duration / avec Robust Scaling / OneHotEncoding pour education / avec optimisation des hyperparamètres
#    """
    if traitement_duration == "Conserver la colonne duration" and traitement_var_num == "Sans RobustScaling" and traitement_education == "Ordinal Encoding" and optimisation_hyperparam == "Non":
        num_test = 1
    elif traitement_duration == "Conserver la colonne duration" and traitement_var_num == "Avec RobustScaling" and traitement_education == "Ordinal Encoding" and optimisation_hyperparam == "Non":
        num_test = 2
    elif traitement_duration == "Supprimer la colonne duration" and traitement_var_num == "Sans RobustScaling" and traitement_education == "Ordinal Encoding" and optimisation_hyperparam == "Non":
        num_test = 3
    elif traitement_duration == "Supprimer la colonne duration" and traitement_var_num == "Avec RobustScaling" and traitement_education == "Ordinal Encoding" and optimisation_hyperparam == "Non":
        num_test = 4
    elif traitement_duration == "Supprimer la colonne duration" and traitement_var_num == "Avec RobustScaling" and traitement_education == "OneHotEncoding" and optimisation_hyperparam == "Non":
        num_test = 5
    elif traitement_duration == "Supprimer la colonne duration" and traitement_var_num == "Avec RobustScaling" and traitement_education == "Ordinal Encoding" and optimisation_hyperparam == "Oui":
        num_test = 6
    elif traitement_duration == "Conserver la colonne duration" and traitement_var_num == "Sans RobustScaling" and traitement_education == "OneHotEncoding" and optimisation_hyperparam == "Non":
        num_test = 7
    elif traitement_duration == "Conserver la colonne duration" and traitement_var_num == "Avec RobustScaling" and traitement_education == "OneHotEncoding" and optimisation_hyperparam == "Non":
        num_test = 8
    elif traitement_duration == "Conserver la colonne duration" and traitement_var_num == "Sans RobustScaling" and traitement_education == "Ordinal Encoding" and optimisation_hyperparam == "Oui":
        num_test = 9
    elif traitement_duration == "Conserver la colonne duration" and traitement_var_num == "Avec RobustScaling" and traitement_education == "Ordinal Encoding" and optimisation_hyperparam == "Oui":
        num_test = 10
    elif traitement_duration == "Conserver la colonne duration" and traitement_var_num == "Sans RobustScaling" and traitement_education == "OneHotEncoding" and optimisation_hyperparam == "Oui":
        num_test = 11
    elif traitement_duration == "Supprimer la colonne duration" and traitement_var_num == "Avec RobustScaling" and traitement_education == "OneHotEncoding" and optimisation_hyperparam == "Oui":
        num_test = 12
    elif traitement_duration == "Supprimer la colonne duration" and traitement_var_num == "Sans RobustScaling" and traitement_education == "Ordinal Encoding" and optimisation_hyperparam == "Oui":
        num_test = 13
    elif traitement_duration == "Supprimer la colonne duration" and traitement_var_num == "Sans RobustScaling" and traitement_education == "OneHotEncoding" and optimisation_hyperparam == "Non":
        num_test = 14
    elif traitement_duration == "Supprimer la colonne duration" and traitement_var_num == "Sans RobustScaling" and traitement_education == "OneHotEncoding" and optimisation_hyperparam == "Oui":
        num_test = 15
    elif traitement_duration == "Conserver la colonne duration" and traitement_var_num == "Avec RobustScaling" and traitement_education == "OneHotEncoding" and optimisation_hyperparam == "Oui":
        num_test = 16
    else:
        print("Test manquant !")
    for model_name, model in models.items():
        test_path = build_ml_path / f"test{num_test}" 
        model_file = test_path / f"ml_test{num_test}_{model_name}_model.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        st.write("#### Modèle testé : " + model_name)
        if optimisation_hyperparam == "Oui" and (model_name in ["Random Forest", "Extreme Gradient Boost", "CatBoost"]):
            params_file = test_path / f"ml_test{num_test}_{model_name}_best_params.pkl"
            with open(params_file, "rb") as f:
                best_params = pickle.load(f)
            best_params_list[model_name] = best_params
            st.write("Meilleurs hyperparamètres pour le modèle ", model_name)
            st.write(best_params)
        else:
            st.write("Aucun hyperparamètre n'a été optimisé pour ce test.")
        y_pred_train = model.predict(X_train_prepro)
        y_pred_test = model.predict(X_test_prepro)
        f1 = f1_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        resultats.append({
            "Modèle": model_name,
            "Précision": precision,
            "Rappel": recall,
            "Score F1": f1
            })
        st.write("Variables les plus importantes du modèle ", model_name)
        nom_var = prepro.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            data_importances = pd.DataFrame({"Variables" : nom_var, "Importance" : importance}).sort_values(by = "Variables", ascending = False)
            top_data_importances = pd.DataFrame({"Variables" : nom_var, "Importance" : importance}).sort_values(by = "Importance", ascending = False)
            st.dataframe(top_data_importances.head(5))
            top_data_importances["Color"] = top_data_importances["Variables"].apply(lambda x: "deep_blue" if x == "numerical__duration" else "light_blue")
            fig = px.bar(top_data_importances.head(5), x="Variables", y="Importance", color="Color", color_discrete_map={"deep_blue": "#005780", "light_blue": "#19D3F3"}, title="Top 5 des variables du modèle " + model_name)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
#            full_path = build_graphs_path / f"graph_var_imp_{model_name}.png"
#            st.image(str(full_path), caption=f"Graphique des variables importantes pour le modèle {model_name}", use_column_width=True)
            for feature, importance in zip(nom_var, importance):
                if feature not in features:
                    features[feature] = {}
                features[feature][model_name] = importance
        elif model_name == "Logistic Regression":
#           coef = model.coef_[0]
            coef = model.coef_
            if coef.ndim == 1:  # Si c'est un tableau 1D (ce qui arrive parfois en classification binaire)
                coef = coef.reshape(1, -1)
            coef = coef[0]
            data_importances = pd.DataFrame({"Variables": nom_var, "Coefficient": coef})
            data_importances["Importance"] = data_importances["Coefficient"].abs()
            data_importances = data_importances.sort_values(by = "Variables", ascending = False)
            top_data_importances = data_importances.sort_values(by = "Importance", ascending = False)
            st.dataframe(top_data_importances.head(5))
            top_data_importances["Color"] = top_data_importances["Variables"].apply(lambda x: "deep_blue" if x == "numerical__duration" else "light_blue")
            fig = px.bar(top_data_importances.head(5), x="Variables", y="Importance", color="Color", color_discrete_map={"deep_blue": "#005780", "light_blue": "#19D3F3"}, title="Top 5 des variables du modèle " + model_name)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
#            full_path = build_graphs_path / f"graph_var_imp_{model_name}.png"
#            st.image(str(full_path), caption=f"Graphique des variables importantes pour le modèle {model_name}", use_column_width=True)
            for feature, importance in zip(nom_var, data_importances["Importance"]):
                if feature not in features:
                    features[feature] = {}
                features[feature][model_name] = importance
        else:
            st.write("Ce modèle ne possède pas d'attribut feature_importances_ ou coef_")
#       Interprétabilité avec SHAP
        if model_name in ["Extreme Gradient Boost", "CatBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_prepro)
            mean_shap_values = pd.DataFrame(shap_values, columns=nom_var).abs().mean()
#           explainer = shap.Explainer(model, X_train_prepro)
#           shap_values = explainer(X_test_prepro)
#           mean_shap_values = pd.DataFrame(shap_values.values, columns=nom_var).abs().mean()
            top_5_shap = mean_shap_values.nlargest(5)
            top_shap_importances = pd.DataFrame({'Variables': top_5_shap.index, 'Importance SHAP': top_5_shap.values})
            st.write("Interprétabilité avec SHAP")
            st.dataframe(top_shap_importances)
            top_shap_importances["Color"] = top_shap_importances["Variables"].apply(lambda x: "deep_blue" if x == "numerical__duration" else "light_blue")
            fig = px.bar(top_shap_importances, x="Variables", y="Importance SHAP", color="Color", color_discrete_map={"deep_blue": "#005780", "light_blue": "#19D3F3"}, title="Interprétabilité avec SHAP - Top 5 des variables du modèle " + model_name)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
#            full_path = build_graphs_path / f"graph_var_shap_{model_name}.png"
#            st.image(str(full_path), caption=f"Graphique de l'interprétabilité des variables pour le modèle {model_name}", use_column_width=True)
        else:
            st.write("L'interprétabilité n'a pas été réalisée pour ce test.")
    st.write("#### Récapitulatif des meilleurs hyperparamètres")
    if optimisation_hyperparam == "Oui":
        for model_name, params in best_params_list.items():
            st.write(f"{model_name}: {params}")
    else:
        st.write("Aucun hyperparamètre n'a été optimisé pour ces tests.")
    recap_resultats = pd.DataFrame(resultats)
    st.write("#### Récapitulatif des performances des différents modèles")
    st.write("**(selon les paramètres choisis précédemment)**")
    st.dataframe(recap_resultats)
    #st.write("#### Importances des variables pour chacun des modèles")
    #st.write("**(selon les paramètres choisis précédemment)**")
    #recap_importances = pd.DataFrame(features).T  # Transposition pour avoir les variables en ligne
    #st.table(recap_importances)


df_pred = pd.DataFrame(columns=["age","balance","duration","campaign","pdays","previous", "job", "marital", "education","default","housing","loan","day","month","poutcome"])

traductions = {
    "oui": "yes",
    "non": "no",
    "célibataire": "single",
    "marié(e)": "married",
    "divorcé(e)": "divorced",
    "primaire": "primary",
    "secondaire": "secondary",
    "supérieur": "tertiary",
    "employé de bureau": "admin.", 
    "technicien": "technician",
    "service à la personne": "services",
    "manager": "management",
    "ouvrier": "blue-collar",
    "retraité": "retired", 
    "étudiant": "student",
    "sans emploi": "unemployed",
    "entrepreneur": "entrepreneur",
    "personnel de ménage": "housemaid",
    "indépendant": "self-employed",
    "succès": "success",
    "échec": "failure",
    "autre": "other",
    }

def traduction_anglais(francais):
    if isinstance(francais, str):
        return traductions.get(francais, francais)  # Traduit si le mot est dans le dictionnaire
    return francais

if page == outil:
    st.header(outil)
    st.write("Renseignez le profil de votre prospect :")
# Rappel :
# Options : avec ou sans duration
# Les autres paramètres sont figés de la manière suivante :
# avec Robust Scaling / OneHotEncoding pour education / sans optimisation des hyperparamètres 
# > test 8 (avec duration) et test 5 (sans duration) 
    choix_age = st.slider("Quel est son âge ?", 18, 95)
    choix_marital = st.selectbox("Quelle est son statut marital ?", ("célibataire", "marié(e)", "divorcé(e)"))
    choix_job = st.selectbox("Quel est son métier ?", ("employé de bureau", "technicien", "service à la personne", "manager", "ouvrier", "retraité", "étudiant", "sans emploi", "entrepreneur", "personnel de ménage", "indépendant"))
    choix_education = st.selectbox("Quelle est son niveau d'études ?", ("primaire", "secondaire", "supérieur"))
    choix_default = st.radio("Est-il/elle en situation de défaut de paiement ?", ("oui", "non"))
    choix_balance = st.number_input("Quel est son solde bancaire ?")
    choix_housing = st.radio("Bénéficie-t-il/elle d'un crédit immobilier ?", ("oui", "non"))
    choix_loan = st.radio("Bénéficie-t-il/elle d'un crédit à la consommation ?", ("oui", "non"))
    choix_date_contact_prevu = st.date_input("A quelle date prévoyez-vous de contacter le prospect ?")
    if st.checkbox("Je souhaite renseigner une durée d'appel prévisionnelle"):
        choix_duration = st.slider("Durée prévue du contact téléphonique (en minutes) ?", 0, 60)
        df_pred.loc[0,"duration"] = choix_duration*60
    else:
        df_pred.loc[0,"duration"] = "unknown"
    df_pred.loc[0,"age"] = choix_age
    df_pred.loc[0,"marital"] = choix_marital
    df_pred.loc[0,"job"] = choix_job
    df_pred.loc[0,"education"] = choix_education
    df_pred.loc[0,"default"] = choix_default
    df_pred.loc[0,"balance"] = choix_balance
    df_pred.loc[0,"housing"] = choix_housing
    df_pred.loc[0,"loan"] = choix_loan
    df_pred.loc[0,"day"] = choix_date_contact_prevu.day
    mois_texte = {1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"}
    df_pred.loc[0,"month"] = mois_texte[choix_date_contact_prevu.month]
    choix_contact = st.radio("Le prospect a-t-il/elle déjà été contacté auparavant ?", ("oui, uniquement pour la campagne actuelle", "oui, uniquement pour la campagne précédente", "oui, pour les 2 campagnes", "non"))
    if choix_contact == "non":
        df_pred.loc[0,"campaign"] = 1
        df_pred.loc[0,"pdays"] = -1
        df_pred.loc[0,"previous"] = 0
        df_pred.loc[0,"poutcome"] = "unknown"
    elif choix_contact == "oui, uniquement pour la campagne actuelle":
        choix_campaign = st.slider("Combien de fois le prospect a-t-il été contacté pour cette campagne (y compris le contact prévu prochainement) ?", 2, 10)
        df_pred.loc[0,"campaign"] = choix_campaign
        df_pred.loc[0,"pdays"] = -1
        df_pred.loc[0,"previous"] = 0
        df_pred.loc[0,"poutcome"] = "unknown"
    elif choix_contact == "oui, uniquement pour la campagne précédente":
        choix_previous = st.slider("Combien de fois le prospect a-t-il été contacté pour une campagne précédente ?", 1, 60)
        choix_date_contact_prec = st.date_input("Quelle était la date de votre dernier contact pour la campagne précédente ?") 
        choix_poutcome = st.selectbox("Quel a été le résultat de la campagne précédente ?", ("succès", "échec", "autre"))
        df_pred.loc[0,"campaign"] = 1
        df_pred.loc[0,"pdays"] = (choix_date_contact_prevu - choix_date_contact_prec).days
        df_pred.loc[0,"previous"] = choix_previous
        df_pred.loc[0,"poutcome"] = choix_poutcome
    elif choix_contact == "oui, pour les 2 campagnes":
        choix_campaign = st.slider("Combien de fois le prospect a-t-il été contacté pour cette campagne (y compris le contact prévu prochainement) ?", 2, 10)
        choix_previous = st.slider("Combien de fois le prospect a-t-il été contacté pour une campagne précédente ?", 1, 60)
        choix_date_contact_prec = st.date_input("Quelle était la date de votre dernier contact pour la campagne précédente ?") 
        choix_poutcome = st.selectbox("Quel a été le résultat de la campagne précédente ?", ("succès", "échec", "autre"))
        df_pred.loc[0,"campaign"] = choix_campaign
        df_pred.loc[0,"pdays"] = (choix_date_contact_prevu - choix_date_contact_prec).days
        df_pred.loc[0,"previous"] = choix_previous
        df_pred.loc[0,"poutcome"] = choix_poutcome
    df_pred = df_pred.applymap(traduction_anglais)
    model_name = "CatBoost"
    if df_pred["duration"][0] == "unknown":
        num_test = 5
        df = df_bank_2
        df_pred = df_pred.drop("duration", axis=1)
        var_num = ["age","balance","campaign","pdays","previous"]
    else:
        num_test = 8
        df = df_bank_1
        var_num = ["age","balance","duration","campaign","pdays","previous"]
    var_cat_for_ohe = ["job","marital", "education","default","housing","loan","day","month","poutcome"]
    data = df.drop("deposit", axis = 1)
    target = df["deposit"]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.30, random_state = 88)
    prepro = ColumnTransformer(transformers = [("numerical", num_scaler, var_num), ("categorical_ohe", ohe, var_cat_for_ohe)])
    X_train_prepro = prepro.fit_transform(X_train)
    y_train = le.fit_transform(y_train)

    df_pred_prepro = prepro.transform(df_pred)

    test_path = build_ml_path / f"test{num_test}" 
    model_file = test_path / f"ml_test{num_test}_{model_name}_model.pkl"
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Prédiction
    y_pred = model.predict(df_pred_prepro)
    y_pred_proba = model.predict_proba(df_pred_prepro)[:,1]
    y_pred_proba_pourcentage = round(y_pred_proba[0]*100, 1)

    # Affichage des résultats
    st.write("##### Ce prospect est-il susceptible de souscrire un dépôt à terme lors de cette campagne ?")

    if y_pred[0] == 1:
        if y_pred_proba_pourcentage > 70:
            st.markdown("""
                        <h1 style='color: green; text-align: center;'>
                        Oui
                        </h1>
                        """,
                        unsafe_allow_html=True
                        )
            st.write("Ce prospect présente une probabilité de ", y_pred_proba_pourcentage,"% de souscrire un dépôt à terme.")
        else:
            st.markdown("""
                        <h1 style='color: orange; text-align: center;'>
                        Oui
                        </h1>
                        """,
                        unsafe_allow_html=True
                        )
            st.write("Mais attention, ce prospect ne présente qu'une probabilité de ", y_pred_proba_pourcentage,"% de souscrire un dépôt à terme.")
    else:
        st.markdown("""
                    <h1 style='color: red; text-align: center;'>
                    Non
                    </h1>
                    """,
                    unsafe_allow_html=True
                    )
        st.write("Ce prospect ne présente qu'une probabilité de ", y_pred_proba_pourcentage,"% de souscrire un dépôt à terme.")



if page == conclusion:
    st.header(conclusion)
    st.markdown(txt_conclusion_generale)

