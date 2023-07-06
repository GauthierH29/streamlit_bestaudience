import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from st_files_connection import FilesConnection


# Create connection object and retrieve file contents.
# Specify input format is a csv and to cache the result for 600 seconds.
conn = st.experimental_connection('gcs', type=FilesConnection)
df = conn.read("bucket-test-bestaudience/data_base_le_wagon.csv", input_format="csv", ttl=600, sep=";")


# Barre de navigation latérale
st.sidebar.title("Best Audience")

# Personnalisation de la mise en page
st.sidebar.markdown("---")

# Options de navigation
pages = ["Profil", "Data Analysis", "Product Recomendation", "Audience Recomendation"]
selected_page = st.sidebar.radio("Navigation", pages)

# Création de la session pour stocker les données
session_state = st.session_state

# Affichage des différentes pages
if selected_page == "Profil":
    # Page Profil
    st.title("Profil")

    # Utilisation de la grille CSS
    col1, col2 = st.columns([1, 1])  # Division en 2 colonnes de largeur égale

    # Colonne de gauche : Écrivez votre nom
    with col1:
        st.subheader("Enter your Name")
        name = st.text_input("Name")
        if name:
            st.markdown(f"<h1>Hello {name} !</h1>", unsafe_allow_html=True)
            st.success("You can drop your CSV file right for the best comercial analysis !")
        else:
            st.warning("Please enter your name.")

    # Colonne de droite : Déposez le fichier
    with col2:
        st.subheader("File Uploader")
        csv_file = st.file_uploader("Upload a CSV", type="csv")
        if csv_file is not None:
            # Sauvegarde du fichier chargé dans la session
            session_state['data'] = pd.read_csv(csv_file)
            st.write("File uploaded successfully !")
        else:
            session_state['data'] = df
# Code pour la page Profil ici

elif selected_page == "Data Analysis":
    plt.style.use('seaborn-whitegrid')

    # Titre de la page
    st.title("Data Analysis")
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
            commande_type_counts = df['Produit - Forme'].value_counts()
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

elif selected_page == "Product Recomendation":
    # Page Product Recommendation
    st.title("Product Recomendation")

    # Vérification si des données ont été chargées dans la session
    if 'data' in session_state:
        data = session_state['data']

        # Sélection des clients
        st.subheader("Sélectionnez les clients")

        # Liste des clients
        client_list = data["Client - ID"].unique().tolist()

        # Sélection des clients
        selected_clients = st.multiselect("Clients", options=client_list, default=[], help="Select clients")

        # Vérification si des clients ont été sélectionnés
        if len(selected_clients) > 0:
            for client_id in selected_clients:
                # Obtention des clients similaires pour chaque client sélectionné
                # à compléter
                similar_customers = data['Client - ID']

                # Obtention des produits achetés pour chaque client sélectionné
                # à compléter
                products_purchased = data['Produit - Forme']

                # Affichage des résultats
                st.write(f"Client ID: {client_id}")
                st.write("Top 10 clients similaires:")
                st.table(similar_customers[:10] )
                st.write("Top 5 produits achetés:")
                st.table(products_purchased[:5])
        else:
            st.write("Aucun client sélectionné.")

    else:
        st.warning("Veuillez charger un fichier CSV depuis la page Profil.")
# Code pour la page Product Recommendation ici

elif selected_page == "Audience Recomendation":
    # Page Audience Recommendation
    st.title("Audience Recomendation")

    # Vérification si des données ont été chargées dans la session
    if 'data' in session_state:
        data = session_state['data']

        # Affichage du DataFrame
        st.subheader("Données brutes")
        st.write(data)

        # Description statistique du DataFrame
        st.subheader("Description statistique")
        st.write(data.describe())

        # Visualisation des clusters
        st.subheader("Visualisation des clusters")

        # Sélection des numéros de clusters
        cluster_numbers = [1,2,3,7,8,13]

        if len(cluster_numbers) > 0:
            # Filtrage des données pour les clusters sélectionnés
            filtered_data = data

            # Affichage du DataFrame filtré
            st.write(filtered_data)

            # Visualisation des clusters
            # à compléter
            #visualize_clusters(filtered_data, cluster_numbers)
        else:
            st.write("Aucun cluster sélectionné.")

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
    name = st.text_input("Nom")

    # Champ Email
    email = st.text_input("Email")

    # Champ Message
    message = st.text_area("Message")

    # Bouton d'envoi du formulaire
    if st.button("Envoyer"):
        # Validation et traitement des données du formulaire
        if not name or not email or not message:
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
