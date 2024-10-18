from django.contrib import admin
from django.urls import path, include
from finance_app import views  # Import the views from finance_app

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),  # Add this line for the root URL
    path('finance/', include('finance_app.urls')),
]
