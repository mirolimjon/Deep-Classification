from django.contrib import admin
from .models import Herb, Images, Review, Userhistory
# Register your models here.


class HerbImagesAdmin(admin.TabularInline):
    model = Images

class HerbAdmin(admin.ModelAdmin):
    inlines = [HerbImagesAdmin]
    list_display = ['id', 'content_images', 'title']

admin.site.register(Herb, HerbAdmin)


class ReviewAdmin(admin.ModelAdmin):
    list_display = ['user', 'herb', 'rating', 'created']
admin.site.register(Review, ReviewAdmin)

class HistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'herb', 'created']
admin.site.register(Userhistory, HistoryAdmin)