from django.urls import path

from aws_textract.views import get_textract, get_textract2

urlpatterns = [
    path('omid/', get_textract),
    path('amirh/', get_textract2)
]
