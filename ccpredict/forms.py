from django import forms
from datetime import datetime

class SaisieCommandeForm(forms.Form):
    maintenant = datetime(2011,12,10)
    Code_Client = forms.IntegerField(label="Code client Datazon", required = True, min_value = 1, max_value = 99999)
    Date_Cde = forms.DateField(label="Date de la commande", initial=maintenant, required = True,)
    NbArticles = forms.IntegerField(label= "Nb d'articles", required = True, min_value=1)
    NbArticlesDiff = forms.IntegerField(label= "Nb d'articles différents", required = True, min_value=1)
    Mnt_Cde = forms.DecimalField(label= "Montant Commande", required = True, min_value=0)

    Prediction = forms.BooleanField(label="Prediction du type de client ==> ", required = False, disabled= True)
    Comment_Prediction = forms.BooleanField(label="Commentaire sur la prediction", required = False, disabled= True)
    def __str__(self):
        return self.titre
