from django.http import HttpResponse
from django.shortcuts import render
import torch
from gramformer import Gramformer

def home(request):
    return render(request, 'home.html')
def result(request):
    
    gf_inference = torch.load(r'D:\mysite\gf.pth')
    
    aa=str(request.GET['pclass'])
    influent_sentences = []
    op=[]
    op1=[]
    context={}
    influent_sentences.append(aa)  

    for influent_sentence in influent_sentences:
        corrected_sentence = gf_inference.correct(influent_sentence)
        # print("[Input] ", influent_sentence)
        op.append(corrected_sentence[0])
        op1.append(influent_sentence)
        
        
        context['h']=influent_sentence
        context['o']=corrected_sentence[0]

        # {'a1':op},{'a2':aa}


    



    return render(request, 'result.html',context)
