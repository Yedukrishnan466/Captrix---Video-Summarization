from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.welcome,name='welcome'),
    path('home/', views.home, name='home'),
    path('about/', views.about,name='about'),
    path('contact/', views.contact,name='contact'),
    path('login/', views.login,name='login'),
    path('signup/', views.signup,name='signup'),
    path('tnc/', views.tnc,name='tnc'),
    path('pr/', views.pr,name='pr'),
    path('upload/', views.upload_video, name='upload_video'),
    path('process-video/', views.process_video, name='process_video'),
    path('check-status/<str:video_id>/', views.check_status),
    path('logout/', views.logout_view, name='logout'),
    path('summaries/', views.your_summaries, name='your_summaries'),
    path('features/', views.features, name='features'),
    path('w_about/', views.w_about, name='w_about')
]