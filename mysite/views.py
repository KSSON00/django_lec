from django.shortcuts import render
from.models import MainConten
from django.shortcuts import get_object_or_404, render
from .models import MainContent

def index(request, content_id):

    content_list= get_object_or_404(MainContent, pk=content_id)
    context={'content_list':content_list}
    return render(request,'mysite/content_list.html',context)

