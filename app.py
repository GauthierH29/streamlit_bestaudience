import streamlit as st
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time
from dataviz import visualize_clusters,visualize_cluster_products
import requests

from sklearn.preprocessing import MinMaxScaler
from st_files_connection import FilesConnection

#####################################################API CALL#######################################################

# Create connection object and retrieve file contents.
# Specify input format is a csv and to cache the result for 600 seconds.
conn = st.experimental_connection('gcs', type=FilesConnection)
df = conn.read("bucket-test-bestaudience/data_base_le_wagon.csv", input_format="csv", ttl=600, sep=";")
df_analyse_basics = conn.read("bucket-test-bestaudience/kmeans_basics_analyse.csv", input_format="csv", ttl=600)
df_analyse_pca = conn.read("bucket-test-bestaudience/kmeans_pca_analyse.csv", input_format="csv", ttl=600)

#####################################################API CALL#######################################################

# Fonction pour faire une requête GET à l'API de prédiction K-means
def get_kmeans_prediction(nb_k):
    api_url = "http://127.0.0.1:8000/kmean/predict"
    response = requests.get(api_url, params={"nb_k": nb_k})
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Fonction pour faire une requête GET à l'API de prédiction K-means avec PCA
def get_kmeans_pca_prediction(nb_k):
    api_url = "http://127.0.0.1:8000/kmean_pca/predict"
    response = requests.get(api_url, params={"nb_k": nb_k})
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Fonction pour faire une requête GET à l'API de recommandation de produits
def get_product_recommendation(user_ids, top_n_similar, top_n_products):
    api_url = "http://127.0.0.1:8000/recommend/predict"
    response = requests.get(api_url, params={"user_ids": user_ids, "top_n_similar": top_n_similar, "top_n_products": top_n_products})
    if response.status_code == 200:
        return response.json()
    else:
        return None

#####################################################API CALL#######################################################

# Barre de navigation latérale
#st.sidebar.title("Best Audience")
st.sidebar.markdown('<h1 style="color: #ff6555ff;">Best Audience</h1>', unsafe_allow_html=True)

# Personnalisation de la mise en page
st.sidebar.markdown("---")

# Options de navigation
pages = ["Profil - Import", "Data Analyses", "Recommandation Produit", "Segmentation Audience"]
selected_page = st.sidebar.radio("Navigation", pages)

# Création de la session pour stocker les données
session_state = st.session_state

# Affichage des différentes pages
if selected_page == "Profil - Import":
    # Page Profil
    #st.title("Profil")
    st.markdown("<h1 style='color: #ff6555ff;'>Profil - Import</h1>", unsafe_allow_html=True)

    # Utilisation de la grille CSS
    col1, col2 = st.columns([1, 1])  # Division en 2 colonnes de largeur égale

    # Colonne de gauche : Écrivez votre nom
    with col1:
        st.subheader("Entrez votre nom")
        name = st.text_input("Nom")
        if name:
            st.markdown(f"<h1>Bonjour {name} !</h1>", unsafe_allow_html=True)
            st.success("Bienvenue sur Best Audience !")
            st.write("Nous proposons des outils d'analyse commerciale puissants pour vous aider à prendre des décisions basées sur les données.")
        else:
            st.warning("Veuillez entrer votre nom.")
            st.empty()  # Supprime les messages précédents

    # Colonne de droite : Déposez le fichier
    with col2:
        st.subheader("Téléchargez un fichier CSV")
        st.write("Téléchargez un fichier CSV pour commencer")
        st.write("Nos algorithmes d'analyse révéleront des informations à partir de vos données.")
        csv_file = st.file_uploader("Télécharger un fichier CSV", type="csv")
        if csv_file is not None:
            st.success("Fichier téléchargé avec succès !")
            st.write("Nous traiterons vos données et générerons des visualisations pertinentes.")
        else:
            session_state['data'] = df
            session_state['df_analyse_basics']=df_analyse_basics
            session_state['df_analyse_pca']=df_analyse_pca

    # Bouton de validation
    if st.button("Démarrer l'analyse"):
        with st.spinner("Vos données sont en cours de traitement..."):
            time.sleep(2)  # Simulation d'un traitement en cours
            st.success("Analyse terminée !")
            st.write("Vos données ont été prétraitées et analysées.")
            st.write("Nous vous fournirons des informations adaptées à vos enjeux business.")
            st.empty()  # Supprime les messages précédents

# Code pour la page Profil ici

elif selected_page == "Data Analyses":
    plt.style.use('seaborn-whitegrid')

    # Titre de la page
    #st.title("Data Analysis")
    st.markdown("<h1 style='color: #ff6555ff;'>Data Analyses</h1>", unsafe_allow_html=True)
    row0_spacer1, row0_1, row0_spacer2, row0_spacer3 = st.columns((.1, 2.3, .1, .1))
    row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
    #intro
    with row3_1:
        st.markdown("Vous trouverez ici un aperçu rapide du tableau de données fourni dans l'onglet Analyse de profil. Il est toujours important de comprendre vos données avant de vous lancer dans l'apprentissage automatique ou toute autre analyse.😉")

    # Vérification si des données ont été chargées dans la session
    if 'data' in session_state:
        data = session_state['data']

        # Affichage des données
        st.header("Aperçu des données")
        with row3_1:
            st.markdown("")
        see_data = st.expander('Vous pouvez cliquer ici pour voir les données brutes 👉')
        with see_data:
            st.dataframe(data=data.reset_index(drop=True))
        st.text('')

        # Analyse exploratoire des données
        st.header("Analyse exploratoire des données")

        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        st.write(data.describe())

        # Visualisation des données
        st.header("Visualisation des données")

        # Choix du graphique à afficher
        selected_chart = st.selectbox("Sélectionnez le type de graphique :", ["Graphique - Date de naissance par date", "Diagramme circulaire des genres", "Diagramme Ventes par produit", "Diagramme CA par date", "Diagramme répartition des sous catégories", "Tendances de création de compte"])

        if selected_chart == "Graphique - Date de naissance par date":
            # Exemple de visualisation : Histogramme
            st.subheader("Histogramme des valeurs de la colonne 'Age'")
            data['Client - Date de Naissance'] = pd.to_datetime(data['Client - Date de Naissance'], errors='coerce')
            fig, ax = plt.subplots(facecolor='white')
            ax.hist(data['Client - Date de Naissance'].dt.year, color='steelblue')
            st.pyplot(fig)

        elif selected_chart == "Diagramme circulaire des genres":
            # Exemple de visualisation : Diagramme circulaire
            st.subheader("Diagramme circulaire des catégories de la colonne 'Sexe'")
            counts = data['Client - Civilité'].value_counts()
            fig, ax = plt.subplots(facecolor='white')
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
            ax.axis('equal')  # Pour s'assurer que le diagramme est circulaire
            st.pyplot(fig)

        elif selected_chart == "Diagramme Ventes par produit":
            # Exemple de visualisation : Diagramme en barres
            fig, ax = plt.subplots()
            sns.countplot(data=data, x='Produit - Catégorie')
            plt.xlabel('Produit')
            plt.ylabel('Nombre de ventes')
            plt.title("Répartition des ventes par produit")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif selected_chart == "Diagramme CA par date":
            data['DAY'] = pd.to_datetime(data['DAY'])

            # Extraire le mois à partir de la colonne 'Date'
            data['Mois'] = data['DAY'].dt.to_period('M')
            data['CA HT'] = data['CA HT'].str.replace(',', '.').astype(float)
            ca_par_date = data.groupby('Mois')['CA HT'].sum().reset_index()

            fig, ax = plt.subplots()
            sns.barplot(data=ca_par_date, x='Mois', y='CA HT')
            plt.xlabel('Date')
            plt.ylabel('Chiffre d\'affaires')
            plt.title('Chiffre d\'affaires par date')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif selected_chart == "Diagramme répartition des sous catégories":
            fig, ax = plt.subplots()
            commande_type_counts = data['Produit - Forme'].value_counts()
            commande_type_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
            plt.title("Répartition des Catégorie")
            plt.ylabel("")
            st.pyplot(fig)

        elif selected_chart == "Tendances de création de compte":
            df['Client - Mois de Création'] = pd.to_datetime(df['Client - Mois de Création'], format='%Y%m')
            df['Client - Mois de la première commande'] = pd.to_datetime(df['Client - Mois de la première commande'], format='%Y%m')
            df['Client - Mois de la dernière commande'] = pd.to_datetime(df['Client - Mois de la dernière commande'], format='%Y%m')

            fig, ax = plt.subplots()
            df['Client - Mois de Création'].dt.to_period('M').value_counts().sort_index().plot(kind='line', figsize=(10, 6))
            plt.title("Tendances de création de compte")
            plt.xlabel("Mois")
            plt.ylabel("Nombre de comptes créés")
            st.pyplot(fig)

    else:
        st.warning("Veuillez charger un fichier CSV depuis la page Profil.")

# Code pour la page Data Analysis ici

elif selected_page == "Recommandation Produit":
    # Page Product Recommendation
    #st.title("Product Recomendation")
    st.markdown("<h1 style='color: #ff6555ff;'>Recommandation Produit</h1>", unsafe_allow_html=True)

    if 'data' in session_state:
        data = session_state['data']

        # Sélection des clients
        st.subheader("Sélectionnez les clients")

        # Liste des clients
        client_list = data["Client - ID"].unique().tolist()

        # Sélection des clients
        selected_clients=st.text_input("Entrez les ID clients sous le format suivant : | séparés par une virgule | (ex: CLT91838,CLT32918,CLT94868,CLT20208,CLT81083) pour lesquels vous souhaitez avoir une recommandation produit")

        # Vérification si des clients ont été sélectionnés
        if len(selected_clients) > 0:

            # Sélection du nombre de clients similaires et de produits recommandés
            top_n_similar = st.number_input("Choix du nombre de clients similaires sur lequel vous souhaitez baser votre recommandation :", min_value=1, value=10, step=1)
            top_n_products = st.number_input("Choix du nombre de produits recommandés par clients similaires sélectionnés sur lequel vous souhaitez vous baser :", min_value=1, value=5, step=1)

            selected_chart = st.selectbox("Sélectionnez un client pour voir  :", selected_clients.split(","))

            # Recommandation de produits
            product_recommendation = get_product_recommendation(selected_clients, top_n_similar, top_n_products)
            if product_recommendation is not None:
                df_recommendations = {}
                for client_id, recommendations in product_recommendation.items():
                    # Création du DataFrame pour le tableau
                    df_recommendations[str(client_id)] = pd.DataFrame(recommendations, columns=["Produit", "Score"])

                # Affichage du tableau
                if selected_chart in df_recommendations.keys() :
                    st.write(df_recommendations[str(client_id)])

                #show_recommendations = st.checkbox("Afficher la liste complète de recommandations")
                #if show_recommendations:
                    #st.write(df_recommendations)


        else:
            st.write("Aucun client sélectionné.")
    else:
        st.warning("Veuillez charger un fichier CSV depuis la page Profil.")


elif selected_page == "Segmentation Audience":
    # Page Audience Recommendation
    #st.title("Audience Recomendation")
    st.markdown("<h1 style='color: #ff6555ff;'>Segmentation Audience</h1>", unsafe_allow_html=True)
    pages_audience = ["pca + kmeans","kmeans"]
    selected_page_audience = st.radio("Modèles", pages_audience)

    if selected_page_audience == "kmeans":
        st.markdown('___')
        st.subheader('Modèle Kmeans')
        nb_k = st.slider('Combien de clusters ?', 1, 30)
        labels=get_kmeans_prediction(nb_k)

        # Vérification si des données ont été chargées dans la session
        if 'data' in session_state:
            data = session_state['data']
            if labels is not None:
                #labels = labels.rename(columns={'labels': 'Cluster Label'})
                labels = pd.DataFrame.from_dict(labels).rename(columns={'labels': 'Cluster Label'})
                df_def = pd.concat([df_analyse_basics,labels],axis=1)

                # Affichage des données
                st.subheader("Aperçu des données labélisées")
                with st.expander("Cliquez pour voir :"):
                    st.write(df_def)

                # Description statistique du DataFrame
                st.subheader("Description statistique")
                with st.expander("Cliquez pour voir :"):
                    st.write(df_def.describe())

                # Visualisation des clusters
                st.subheader("Visualisation des clusters")

                # Sélection des numéros de clusters
                cluster_numbers = sorted(labels['Cluster Label'].unique())
                for i in range(len(cluster_numbers)):
                    cluster_numbers[i]+=1

                # Affichez les cases à cocher
                cluster_numbers=st.multiselect("Selectionnez les clusters",cluster_numbers)

                with st.expander("Cliquez pour voir les graphiques :"):
                    if len(cluster_numbers) > 0:
                        # Filtrage des données pour les clusters sélectionnés
                        filtered_data = data

                        # Visualisation des clusters
                        # Test méthode avec les clusters choisis
                        visualize_clusters(df_def,cluster_numbers)
                        #visualize_clusters(filtered_data, cluster_numbers)
                    else:
                        st.write("Aucun cluster sélectionné.")
            else:
                st.warning("Veuillez lancer l'api.")

        else:
            st.warning("Veuillez charger un fichier CSV depuis la page Profil.")

    elif selected_page_audience == "pca + kmeans":
        st.markdown('___')
        st.subheader('Modèle pca + kmeans')
        nb_k = st.slider('Combien de clusters ?', 1, 30)
        labels=get_kmeans_pca_prediction(nb_k)

        # Vérification si des données ont été chargées dans la session
        if 'data' in session_state:
            data = session_state['data']
            if labels is not None:
                labels = pd.DataFrame.from_dict(labels).rename(columns={'labels': 'Cluster Label'})
                df_def = pd.concat([df_analyse_pca,labels],axis=1)

                # Affichage des données
                st.subheader("Aperçu des données labélisées")
                with st.expander("Cliquez pour voir :"):
                    st.write(df_def)

                # Description statistique du DataFrame
                st.subheader("Description statistique")
                with st.expander("Cliquez pour voir :"):
                    st.write(df_def.describe())

                # Visualisation des clusters
                st.subheader("Visualisation des clusters")

                # Sélection des numéros de clusters
                cluster_numbers = sorted(labels['Cluster Label'].unique())
                for i in range(len(cluster_numbers)):
                    cluster_numbers[i]+=1

                # Affichez les cases à cocher
                cluster_numbers=st.multiselect("Selectionnez les clusters",cluster_numbers)

                with st.expander("Cliquez pour voir les graphiques :"):
                    if len(cluster_numbers) > 0:
                        # Filtrage des données pour les clusters sélectionnés
                        filtered_data = data

                        #selected_chart = st.selectbox("Sélectionnez le type de graphique :", ["Analyses propriétés utilisateurs", "Analyses propriétés produits"])

                        #if selected_chart == "Graphique - Date de naissance par date":
            # Exemple de visualisation : Histogramme
                            #filtered_data = data
                        visualize_clusters(df_def,cluster_numbers)
                        #elif selected_chart == "Graphique - Date de naissance par date":

                        # Visualisation des clusters
                        # Test méthode avec les clusters choisis
                            #filtered_data = data
                            #visualize_cluster_products(df_def,cluster_numbers)
                        #visualize_clusters(filtered_data, cluster_numbers)
                    else:
                        st.write("Aucun cluster sélectionné.")
            else:
                st.warning("Veuillez lancer l'api.")

        else:
            st.warning("Veuillez charger un fichier CSV depuis la page Profil.")
        # Code pour la page Audience Recommendation ici

# Personnalisation de la mise en page
st.sidebar.markdown("---")

# Bouton Contact
if st.sidebar.button("Contact"):
    st.title("Contact")

    # Formulaire de contact
    st.subheader("Formulaire de contact")

    # Champ Nom
    naming = st.text_input("Nom Prénom")

    # Champ Email
    email = st.text_input("Email")

    # Champ Message
    message = st.text_area("Message")

    # Bouton d'envoi du formulaire
    if st.button("Envoyer"):
        # Validation et traitement des données du formulaire
        if not naming or not email or not message:
            st.error("Veuillez remplir tous les champs du formulaire.")
        else:
            # Envoi du message ou traitement des données
            # (vous pouvez ajouter ici le code pour envoyer le message par e-mail ou le traiter d'une autre manière)
            st.success("Votre message a été envoyé avec succès.")

    # Liens vers des réseaux sociaux
    st.subheader("Réseaux sociaux")
    st.write("Suivez-nous sur git hub pour rester informé(e) :")
    st.write("- [GitHub_Gauthier](https://github.com/GauthierH29)")
    st.write("- [GitHub_Derhen](https://github.com/Tshelb)")
    st.write("- [GitHub_Samih](https://github.com/SamihBACHNOU)")

    # Noms des fondateurs
    st.subheader("Fondateurs")
    st.write("Les fondateurs du site sont :")
    st.write("- Gauthier HAICAULT")
    st.write("- Derhen BOURGEAIS")
    st.write("- Samih BACHNOU")
