from django.contrib import admin
from django.urls import path, include
from finance_app import views  # Import the views from finance_app
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),  # Add this line for the root URL
    path('finance/', include('finance_app.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
