from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class VideoSummary(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video_file_name = models.CharField(max_length=255)
    duration = models.CharField(max_length=20)
    file_size = models.CharField(max_length=20)

    summary = models.TextField()

    audio_file_url = models.URLField(max_length=500)

    image_urls = models.JSONField()      # Store list of image URLs
    captions = models.JSONField()        # Store list of captions

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.video_file_name}"