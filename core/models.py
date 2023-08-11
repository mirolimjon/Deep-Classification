from django.db import models
from shortuuid.django_fields import ShortUUIDField
from django.utils.html import mark_safe
from django.contrib.auth.models import User
from account.models import Profile
# Create your models here.


class Herb(models.Model):
    id = ShortUUIDField(length=8, max_length=16, prefix="herb_", alphabet="abcdefg123456", primary_key=True)
    image = models.ImageField(upload_to="predict/images")
    title = models.CharField(max_length=100)
    body = models.TextField()
    num_views = models.IntegerField(default=0)

    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['-created']

    def content_images(self):
        return mark_safe('<img src="%s" width="50" height="50" />' % (self.image.url))


class Images(models.Model):
    herb = models.ForeignKey(Herb, on_delete=models.CASCADE)
    images = models.ImageField(upload_to='images')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.herb.title
    
    class Meta:
        ordering = ['-created']

    def content_images(self):
        return mark_safe('<img src="%s" width="50" height="50" />' % (self.image.url))




RATING = (
    (1, '★☆☆☆☆'),
    (2, '★★☆☆☆'),
    (3, '★★★☆☆'),
    (4, '★★★★☆'),
    (5, '★★★★★'),
)
class Review(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    herb = models.ForeignKey(Herb, on_delete=models.CASCADE)
    review = models.TextField()
    rating = models.IntegerField(choices=RATING, default=None)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "Content Review"
        ordering = ['-created']

    def __str__(self):
        return self.herb.title
    
    def get_rating(self):
        return self.rating
    
    def get_content_image(self):
        return mark_safe('<img src="%s" width="50" height="50" />' % (self.content.image.url))
    
    def average_rating(self):
        sum_rating = sum(self.rating)
        count = count(self.rating)
        return sum_rating/count

class Userhistory(models.Model):
    # profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    herb = models.ForeignKey(Herb, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user} - {self.herb.title}"