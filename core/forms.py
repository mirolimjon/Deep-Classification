from django import forms
from .models import Review


class CreateReviewForm(forms.ModelForm):
    review = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control w-100'}))
    class Meta:
        model = Review
        fields = ['rating', 'review']