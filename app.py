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
    # Page Data Analysis
    st.title("Data Analysis")

    # Vérification si des données ont été chargées dans la session
    if 'data' in session_state:
        data = session_state['data']

        # Affichage des données
        st.subheader("Aperçu des données")
        st.write(data.head())

        # Analyse exploratoire des données
        st.subheader("Analyse exploratoire des données")

        # Statistiques descriptives
        st.write("Statistiques descriptives :")
        st.write(data.describe())

        # Visualisation des données
        st.write("Visualisation des données :")

        # Exemple de visualisation : Histogramme
        st.write("Histogramme des valeurs de la colonne 'Age'")
        data['Client - Date de Naissance'] = pd.to_datetime(data['Client - Date de Naissance'])
        plt.hist(data['Client - Date de Naissance'].dt.year)
        st.pyplot()

        # Exemple de visualisation : Diagramme circulaire
        st.write("Diagramme circulaire des catégories de la colonne 'Sexe'")
        counts = data['Client - Civilité'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
        st.pyplot()

        # Exemple de visualisation : Diagramme en barres
        st.write("Diagramme en barres des ventes par produit")
        product_sales = data['Produit - Forme'].value_counts()
        sns.barplot(x=product_sales.index, y=product_sales.values)
        plt.xticks(rotation=45)
        st.pyplot()
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
