from django.shortcuts import render, redirect
from .models import Herb, Review, Userhistory
import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
import tensorflow as tf
from django.db.models import Q
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError
from .forms import CreateReviewForm
from django.core.files.storage import FileSystemStorage
from django.db.models import Avg
from django.db.models import Q
from django.contrib import messages
from django.contrib.auth.decorators import login_required
# Create your views here.



labels = []
images = []

filepath = "C:/Users/User/Desktop/Classification/CNN/data/"

# Loop through
for i in os.listdir(filepath):
  for image in os.listdir(filepath + i):
    labels.append(i)
    images.append(filepath + i + "/" + image)  # Gets images location
unique_labels = np.unique(labels)


IMG_SIZE = 224
# Create a function for processing images
def processing_images(filepath):
  image = tf.io.read_file(filepath)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  image = np.expand_dims(image, axis=0)
  return image

class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

# load model
model = tf.keras.models.load_model('CNN/model/20230517-03051684295165-MobileNetV2.h5',
                                   custom_objects = {"KerasLayer": hub.KerasLayer})



def home(request):
    message = ""
    prediction = ""
    fss = CustomFileSystemStorage()
    try:
        image = request.FILES["image"]
        print("Name", image.file)
        _image = fss.save(image.name, image)
        # path = media/jyh_1.jpg"
        path = "C:/Users/User/Desktop/Classification/" + str(settings.MEDIA_ROOT) + "/" + image.name

        image_url = fss.url(_image)
        # image details
        
        print()
        image_processed = processing_images(path)
        
        # prodict
        result = model.predict(image_processed) 


        print("Prediction: " + str(np.argmax(result)))

        print("Similarity: {:.2f} %".format(np.max(result)*100))
        print(f"Max value (Probability prediction) {np.max(result)}")
        print(f"Sum: {np.sum(result)}")
        print(f"Max. index: {np.argmax(result)}")
        print(f"Label: {unique_labels[np.argmax(result)]}")

        prediction = unique_labels[np.argmax(result)] 
        similarity = np.max(result) * 100
        
        herbs = Herb.objects.filter(
            Q(title__icontains=prediction)
        )
        
        return TemplateResponse(
            request,
            "index.html",
            {
                "message": message,
                "image": image,
                "image_url": image_url,
                "prediction": prediction,
                'herbs': herbs,
                'similarity': similarity
            },
        )
    except MultiValueDictKeyError:

        return TemplateResponse(
            request,
            "index.html",
            {"message": "No Image Selected"},
        )


def productsList(request):
    herbs = Herb.objects.all()
    context = {
        'herbs': herbs
    }
    return render(request, 'core/products.html', context)


@login_required(login_url='login')
def productDetail(request, pk):
    herb = Herb.objects.get(id=pk)
    herb.num_views += 1
    herb.save()
    
    # Create user view history
    if request.user.is_authenticated:
        new_history = Userhistory.objects.update_or_create(
            user = request.user,
            herb = herb
        )    
            
    # ###### Review form
    
    # Checking if user already posted comment or not
    make_review = True
    user_review_count = Review.objects.filter(user=request.user, herb=herb).count()
    if user_review_count>0:
        make_review = False
        
        
    forms = CreateReviewForm()
    if request.method == "POST":
        forms = CreateReviewForm(request.POST)
        if forms.is_valid():
            new_review = forms.save(commit=False)
            new_review.user = request.user
            new_review.herb = herb
            new_review.save()
            messages.success(request, "Review added successfully.")
            return redirect('detail', herb.id)
        
    # Reviews
    get_reviews = Review.objects.filter(herb=herb)[:3]
    average_review = Review.objects.filter(herb=herb).aggregate(rating=Avg('rating'))
    reviews = Review.objects.filter(herb=herb).count()
    count_1 = Review.objects.filter(herb=herb, rating=1).count()
    count_2 = Review.objects.filter(herb=herb, rating=2).count()
    count_3 = Review.objects.filter(herb=herb, rating=3).count()
    count_4 = Review.objects.filter(herb=herb, rating=4).count()
    count_5 = Review.objects.filter(herb=herb, rating=5).count()

    rating_1 = 0
    rating_2 = 0
    rating_3 = 0
    rating_4 = 0
    rating_5 = 0

    if count_1 > 0:
        rating_1 = (count_1*100)/reviews
    if count_2 > 0:
        rating_2 = (count_2*100)/reviews
    if count_3 > 0:
        rating_3 = (count_3*100)/reviews
    if count_4 > 0:
        rating_4 = (count_4*100)/reviews
    if count_5 > 0:
        rating_5 = (count_5*100)/reviews
    else:
        print("")
        
    
    
    
    context = {
        'herb': herb,
        'average_review': average_review,
        'rating_1': rating_1,
        'rating_2': rating_2,
        'rating_3': rating_3,
        'rating_4': rating_4,
        'rating_5': rating_5,
        'forms': forms,
        'get_reviews': get_reviews,
        'make_review': make_review,
    }
    return render(request, 'core/product-detail.html', context)



def aboutPage(request):
    return render(request, 'core/about.html')



def search_herb(request):
    query = request.GET.get('q')
    herbs = Herb.objects.filter(Q(title__icontains=query)|
                                Q(body__icontains=query))
                                                
    context = {
        'herbs': herbs
    }
    return render(request, 'core/search-herb.html', context)