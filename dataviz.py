import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def visualize_clusters(df, cluster_numbers):
    # Filtrer le DataFrame en fonction des numéros de cluster spécifiés
    filtered_df = df[df['Cluster Label'].isin(cluster_numbers)]

    # Exemples de visualisations avec Seaborn
    # Histogramme du nombre de commandes
    plt.figure()
    sns.histplot(data=filtered_df, x='N Commandes', hue='Cluster Label', kde=True, bins=10)
    plt.xlabel('Nombre de commandes')
    plt.ylabel('Fréquence')
    plt.title('Comparaison des clusters - Distribution du nombre de commandes')
    #st.write(plt.show())
    st.pyplot(plt.gcf())

    # Diagramme en boîte de l'ancienneté
    plt.figure()
    sns.boxplot(data=filtered_df, x='Cluster Label', y='Ancienneté')
    plt.xlabel('Cluster')
    plt.ylabel('Ancienneté')
    plt.title('Comparaison des clusters - Diagramme en boîte de l\'ancienneté')
    st.pyplot(plt.gcf())

    # Répartition des clients par cluster
    plt.figure()
    cluster_counts = df['Cluster Label'].value_counts()
    sns.histplot(filtered_df, x='Cluster Label', bins=len(cluster_counts))
    plt.xlabel('Cluster')
    plt.ylabel('Nombre de clients')
    plt.title('Répartition des clients par cluster')
    st.pyplot(plt.gcf())

    plt.figure()
    sns.scatterplot(filtered_df, x='age', y='CA Produits HT', hue='Cluster Label', palette='Set2')
    plt.xlabel('Âge')
    plt.ylabel('Montant des ventes')
    plt.title('Relation entre l\'âge et le montant des ventes')
    st.pyplot(plt.gcf())

    plt.figure()
    sns.scatterplot(filtered_df, x='N Commandes', y='CA Produits HT', hue='Cluster Label', palette='Set2')
    plt.xlabel('Nombre de commandes')
    plt.ylabel('Montant des ventes')
    plt.title('Relation entre le nombre de commandes et le montant des ventes')
    st.pyplot(plt.gcf())

    plt.figure()
    sns.boxplot(filtered_df, x='Cluster Label', y='CA Produits HT', palette='Set3')
    plt.xlabel('Cluster')
    plt.ylabel('Montant des ventes (CA Produit HT)')
    plt.title('Montant des ventes par cluster')
    st.pyplot(plt.gcf())

    plt.figure()
    sns.scatterplot(filtered_df, x='age', y='Ancienneté', hue='Cluster Label', palette='Set2')
    plt.xlabel('Âge')
    plt.ylabel('Ancienneté')
    plt.title('Relation entre l\'âge et l\'ancienneté par cluster')
    st.pyplot(plt.gcf())


    clusters = filtered_df['Cluster Label'].unique()
    print(clusters)
    cmap = get_cmap(len(clusters))
    cluster_df = filtered_df[['CA Produits HT','N Commandes','age','Ancienneté','Cluster Label']]
    cluster_df = cluster_df.groupby('Cluster Label').mean()
    scaler = MinMaxScaler()
    cluster_df = pd.DataFrame(scaler.fit_transform(cluster_df), columns=cluster_df.columns)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})

    for i, cluster in enumerate(clusters):
        #cluster_df2 = cluster_df.loc[cluster]
        cluster_df2 = filtered_df[filtered_df['Cluster Label'] == cluster]

        avg_ca = cluster_df2['CA Produits HT'].mean()
        avg_commandes = cluster_df2['N Commandes'].mean()
        avg_age = cluster_df2['age'].mean()
        avg_anciennete = cluster_df2['Ancienneté'].mean()

        features = ['CA Produits HT', 'N Commandes', 'Age', 'Ancienneté']
        values = [avg_ca, avg_commandes, avg_age, avg_anciennete]

        # Calculer les angles pour chaque caractéristique
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()

        # Créer le graphique radar
        ax.fill(angles, values, color=cmap(i), alpha=0.3)  # Remplir le polygone avec une couleur

    ax.set_xticks(angles)  # Définir les positions des étiquettes des axes
    ax.set_xticklabels(features)  # Définir les étiquettes des axes
    ax.set_yticklabels([])  # Masquer les étiquettes des valeurs
    ax.set_title(f'Cluster Map radar')  # Définir le titre du graphique
    ax.grid(True)  # Afficher une grille

    # Afficher le graphique
    st.pyplot(plt.gcf())

def find_optimal_k(data_scaled, max_k=30):
    inertia_values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_scaled)
        inertia_values.append(kmeans.inertia_)

    # pour tracer la elbow courbe
    plt.plot(range(1, max_k + 1), inertia_values, marker='o')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude (Elbow method)')
    plt.show()

    diff = np.diff(inertia_values)
    elbow_index = np.argmax(diff) + 1
    optimal_n = elbow_index + 1
    return optimal_n


def visualize_cluster_products(df, cluster_numbers):
    # Filtrer le DataFrame en fonction des numéros de cluster spécifiés
    filtered_df = df[df['Cluster Label'].isin(cluster_numbers)]
    df_cluster_products = filtered_df.groupby('Cluster Label')[['Produit - Forme_Body',
 'Produit - Forme_Bonnets',
 'Produit - Forme_Boxers',
 'Produit - Forme_Brassière',
 'Produit - Forme_Chaussettes',
 'Produit - Forme_Chaussure de marche',
 'Produit - Forme_Chaussure fitness',
 'Produit - Forme_Chaussure lifestyle',
 'Produit - Forme_Chaussure mode',
 'Produit - Forme_Chaussure running',
 'Produit - Forme_Chemises',
 'Produit - Forme_Collants',
 'Produit - Forme_Combishort',
 'Produit - Forme_Coupe pluie',
 'Produit - Forme_Coupe vent',
 'Produit - Forme_Crème visage',
 'Produit - Forme_Doudoune hiver',
 'Produit - Forme_Doudoune légère',
 'Produit - Forme_Duo de bloomers',
 'Produit - Forme_Débardeur',
 'Produit - Forme_Débardeur classique',
 'Produit - Forme_Electronique',
 'Produit - Forme_Fitness',
 'Produit - Forme_Gants',
 'Produit - Forme_Haut classique',
 'Produit - Forme_Haut de fitness',
 'Produit - Forme_Huile Corps',
 'Produit - Forme_Leggings running',
 'Produit - Forme_Lessive douce pour VETEMENTS délicate',
 'Produit - Forme_Maillot de bain classique',
 'Produit - Forme_Maillot de bain piscine',
 'Produit - Forme_Maillot de bain sport',
 'Produit - Forme_Maillot enfant',
 'Produit - Forme_Manteau',
 'Produit - Forme_Manteau hiver',
 'Produit - Forme_Masque de CHAUSSURES',
 'Produit - Forme_Mitaines',
 'Produit - Forme_Mobilité',
 'Produit - Forme_Moyenne',
 'Produit - Forme_Outdoor',
 'Produit - Forme_Pantacourts',
 'Produit - Forme_Pantalon running',
 'Produit - Forme_Pantalons déperlants',
 'Produit - Forme_Pantalons modulables',
 'Produit - Forme_Peignoirs',
 'Produit - Forme_Petite',
 'Produit - Forme_Piscine et plage',
 'Produit - Forme_Plein air',
 'Produit - Forme_Polaires',
 'Produit - Forme_Polo sport',
 'Produit - Forme_Pull',
 'Produit - Forme_Sac',
 'Produit - Forme_Sac à dos',
 'Produit - Forme_Short',
 'Produit - Forme_Short classique',
 'Produit - Forme_Short classique ',
 'Produit - Forme_Short classique taille haute',
 'Produit - Forme_Short running',
 'Produit - Forme_Slip boxer de bain',
 'Produit - Forme_Sous-gants',
 'Produit - Forme_Sous-vêtements thermique',
 'Produit - Forme_Stick lèvre SPF20',
 'Produit - Forme_String',
 'Produit - Forme_Survêtements',
 'Produit - Forme_Sweat classique',
 'Produit - Forme_Sweat à capuche',
 'Produit - Forme_T-shirt classique',
 'Produit - Forme_T-shirt de running',
 'Produit - Forme_T-shirt manches longues',
 'Produit - Forme_T-shirt manches longues fitness',
 'Produit - Forme_T-shirt manches longues running',
 'Produit - Forme_Une pièce',
 'Produit - Forme_Veste Hiver',
 'Produit - Forme_Veste été',
 'Produit - Forme_Vêtements plage']].sum().reset_index()

# Melt le DataFrame pour avoir une seule colonne "Forme de produit" et une colonne "Fréquence" correspondant aux valeurs
    df_cluster_products_melted = pd.melt(df_cluster_products, id_vars='Cluster Label', var_name='Forme de produit', value_name='Fréquence')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster Label', y='Fréquence', hue='Forme de produit', data=df_cluster_products_melted)
    plt.xlabel('Cluster Label')
    plt.ylabel('Fréquence')
    plt.title('Répartition des produits par Cluster Label')
    plt.legend(bbox_to_anchor=(1.05, 1))
    st.pyplot(plt.gcf())
