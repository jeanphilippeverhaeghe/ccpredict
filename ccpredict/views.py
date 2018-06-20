from django.shortcuts import render

# Create your views here.
def index(request):
    """
    Index
    """
    return render(request, 'index.html')


from .forms import SaisieCommandeForm

def saisie_order(request):

    form = SaisieCommandeForm(request.POST or None)

    if form.is_valid():
        # Ici nous pouvons traiter les données du formulaire
        Code_Client = form.cleaned_data['Code_Client']
        Date_Cde = form.cleaned_data['Date_Cde']
        NbArticles = form.cleaned_data['NbArticles']
        NbArticlesDiff = form.cleaned_data['NbArticlesDiff']
        Mnt_Cde = form.cleaned_data['Mnt_Cde']

        print ("Code Client: " ,Code_Client)
        print ("Date_Cde: ",Date_Cde)
        print ("NbArticles: ",NbArticles)
        print ("NbArticlesDiff: ",NbArticlesDiff)
        print ("Mnt_Cde: ",Mnt_Cde)

        from . import Datazon_Predictor
        pred = Datazon_Predictor.predict(Code_Client, Date_Cde, NbArticles, NbArticlesDiff, Mnt_Cde)
        comment_pred = "Prédiction via Réseux de Neurones"

        form.fields['Prediction'].label="Prédiction de la segmentation ClienT: " + str(pred)
        form.fields['Comment_Prediction'].label="Commentaire sur la prédiction: " + comment_pred

    # Quoiqu'il arrive, on affiche la page du formulaire.
    return render(request, 'saisie_order.html', locals())
