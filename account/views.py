from django.shortcuts import render, redirect
from .models import Profile
from django.contrib.auth import logout, login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from core.models import Userhistory, Review
# Create your views here.

login_required(login_url='login')
def userProfile(request):
    user = request.user
    profile = Profile.objects.filter(user=user)
    # History
    history = Userhistory.objects.filter(user=user)
    # Review
    review = Review.objects.filter(user=user)
    context = {
        'profile': profile,
        'history': history,
        'review': review,
    }
    return render(request, 'account/profile.html', context)


def loginView(request):
    if request.user.is_authenticated:
        messages.info(request, f"You have already logged in")
        return redirect('index')
    
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        
        try:
            user = User.object.filter(username=username).exists()
        except:
            pass
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f"User login succesfully")
            return redirect('home')
        else:
            messages.warning(request, f"User does not exists")
    return render(request, 'account/login.html')


def registerView(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        password2 = request.POST['password2']
        
        if password == password2:
            if User.objects.filter(username=username).exists():
                messages.warning(request, f"Username is already exists.")
            else:
                new_user = User.objects.create_user(
                    username=username,
                    password=password
                )
                new_user.username = new_user.username.lower()
                new_user.save()
                messages.info(request, f"Your account created succesfully.")
                return redirect('login')
        else:
            messages.warning(request, f"Passwords should be same.")
            return redirect('register')
    return render(request, 'account/register.html')

def logoutView(request):
    logout(request)
    return redirect('login')

