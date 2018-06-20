import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn import  neural_network

#pour création jeu de test/entrainement
from sklearn.model_selection import train_test_split, GridSearchCV

def predict(Code_Client, Date_Cde = datetime(2011,12,10), Nb_Art = 1, Nb_Art_Diff = 1, Mnt_Cde = 1):
    """
    Proc to make prediction on the customer cluster after an order received in parameters
    Code_Client: Datazon Customer ID (integer)
    Date_Cde = order date (datetime)
    Nb_Art = number of items in the order (integer)
    Nb_Art_Diff = number of distinct items in the order (integer)
    Mnt_Cde = amount of the order (decimal)

    Return
    prediction: number of the cluster (integer)
    """

    #Lecture du fichier seg_custom2_with_cluster_end_1.2.csv
    seg_custom = pd.read_csv('seg_custom2_with_cluster_end_1.2.csv', sep=",",
                            encoding='utf_8', low_memory=False)

    #Reconstruction du DataFrame avec CustomerID comme Index
    seg_custom.set_index(seg_custom["CustomerID"],inplace=True)
    seg_custom = seg_custom.drop(['CustomerID'], axis=1)

    # Destruction des deux clients très particuliers:
    #Cherchons automatiquement quel est le cluster concerné
    #   (c'est forcément le dernier dans la série renvoyée par value_counts)
    ma_serie = seg_custom['Cluster_KM_V2'].value_counts()
    cluster_a_eliminer = ma_serie.index[-1]

    #Recherche de leurs indexes
    Liste_Index = seg_custom.loc[seg_custom.Cluster_KM_V2 == cluster_a_eliminer].index.values
    Liste_Index = list(Liste_Index)

    #Elimination
    if len(Liste_Index) == 2 :
        seg_custom = seg_custom.drop(Liste_Index)

    #Copie de base du DF
    seg_custom_O = seg_custom.copy()

    ###############################
    #Etape 1: chercher si le client a déjà passé commande
    seg_custom_client = seg_custom.iloc[seg_custom.index.values == Code_Client]
    if seg_custom_client.shape[0] != 0 :
        client_existe=True
        ante_nb_cde = seg_custom_client['Nb_Total_Cdes'].values[0]
        ante_nb_art = seg_custom_client['Nb_Total_Articles'].values[0]
        ante_nb_dates_dist_cdes = seg_custom_client['Nb_Dates_Distinctes_Cdes'].values[0]
        ante_mnt_moyen_cde = seg_custom_client['Mnt_Moyen_Cdes'].values[0]
        ante_nb_moyen_art_cde = seg_custom_client['Nb_Moyen_Article_par_Cdes'].values[0]
        ante_nb_moyen_art_diff_cde = seg_custom_client['Nb_Moyen_Article_Diff_par_Cdes'].values[0]
        ante_nb_jour_depuis_der_cde = seg_custom_client['Nb_Jour_Depuis_Derniere_Cde'].values[0]
        ante_nb_jour_entre_2_cdes = seg_custom_client['Nb_Jour_Entre_2_Cdes'].values[0]
        ante_mnt_total_cdes = seg_custom_client['Mnt_Total_Cdes'].values[0]
        ante_date_der_cde = datetime(int(seg_custom_client['Date_Der_Cde'].values[0][:4]),
                                 int(seg_custom_client['Date_Der_Cde'].values[0][5:7]),
                                 int(seg_custom_client['Date_Der_Cde'].values[0][8:10]),0,0,0 )
    else :
        client_existe=False
        ante_nb_cde = 0
        ante_nb_art = 0
        ante_nb_dates_dist_cdes = 0
        ante_mnt_moyen_cde = 0
        ante_nb_moyen_art_cde = 0
        ante_nb_moyen_art_diff_cde = 0
        ante_nb_jour_depuis_der_cde = 0
        ante_nb_jour_entre_2_cdes = 0
        ante_mnt_total_cdes = 0
        ante_date_der_cde = Date_Cde

    #Etape 2: cumuler aux montants et nombres éventuels antérieurs les nouveaux nombres fournis
    nb_cde = ante_nb_cde + 1
    nb_art = ante_nb_art + Nb_Art

    nb_dates_dist_cdes = ante_nb_dates_dist_cdes + 1

    mnt_moyen_cde = (float(ante_mnt_moyen_cde * ante_nb_cde) + float(Mnt_Cde)) / (ante_nb_cde + 1)
    mnt_total_cdes = float(ante_mnt_total_cdes) + float(Mnt_Cde)

    nb_moyen_art_cde = ((ante_nb_moyen_art_cde * ante_nb_cde) + Nb_Art) / (ante_nb_cde + 1)
    nb_moyen_art_diff_cde = ((ante_nb_moyen_art_diff_cde * ante_nb_cde) + Nb_Art_Diff) / (ante_nb_cde + 1)

    print(Date_Cde)
    print(ante_date_der_cde)

    delais = Date_Cde - datetime.date(ante_date_der_cde)
    nb_jour_depuis_der_cde = delais.days
    nb_jour_entre_2_cdes = ((ante_nb_jour_entre_2_cdes * ante_nb_cde) + delais.days) / (ante_nb_cde + 1)

    #Etape 3: Lancer la prédiction du Cluster
    #Au préalable se créer un dataframe avec nos nouvelles données dont il faut prédire le cluster
    ar = np.array([[nb_cde,nb_art,
                nb_dates_dist_cdes,mnt_moyen_cde,
                nb_moyen_art_cde,nb_moyen_art_diff_cde,
                nb_jour_depuis_der_cde,nb_jour_entre_2_cdes,
                mnt_total_cdes]])
    df = pd.DataFrame(ar, index = [1], columns = ['Nb_Total_Cdes', 'Nb_Total_Articles',
                                              'Nb_Dates_Distinctes_Cdes', 'Mnt_Moyen_Cdes',
                                              'Nb_Moyen_Article_par_Cdes', 'Nb_Moyen_Article_Diff_par_Cdes',
                                              'Nb_Jour_Depuis_Derniere_Cde', 'Nb_Jour_Entre_2_Cdes',
                                              'Mnt_Total_Cdes'])
    #print(df)

    #Puis préprons les données d'entrainement / test
    #Ne gardons que les features intéressantes:
    cols = list(['Nb_Total_Cdes', 'Nb_Total_Articles', 'Nb_Dates_Distinctes_Cdes', 'Mnt_Moyen_Cdes',
                'Nb_Moyen_Article_par_Cdes', 'Nb_Moyen_Article_Diff_par_Cdes',
                'Nb_Jour_Depuis_Derniere_Cde', 'Nb_Jour_Entre_2_Cdes', 'Mnt_Total_Cdes'])
    #Données pour prédire
    x_final = seg_custom_O[cols]

    #Donnée à prédire
    y_final = seg_custom_O.iloc[:,-1] #C'est la dernière colonne du DF

     #Split Training / Test
    x_train, x_test, y_train, y_test = train_test_split(x_final,y_final,test_size = 0.3,random_state = 0) # Do 70/30 split

    #Normalisons les données
    scaler = StandardScaler() # create scaler object
    scaler.fit(x_train) # fit with the training data ONLY
    x_train = scaler.transform(x_train) # Transform the data
    x_test = scaler.transform(x_test) # Transform the data

    #Instancions un réseau de neurones
    #rint("Neural Network Classifier: (Multi Layer Perceptron)")
    lr = neural_network.MLPClassifier(solver= 'adam',
                                  hidden_layer_sizes = (9,6),
                                  activation = 'identity')
    lr.fit(x_train,y_train)

    #Prédiction
    prediction = lr.predict(df)

    return prediction[0]
