from django.contrib import admin
from django.urls import path, include
from finance_app import views
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('finance/', include('finance_app.urls')),
    
    # Add this to serve static files
    path('bull.jpg', serve, {'document_root': settings.STATIC_ROOT, 'path': 'bull.jpg'}),
    path('favicon.ico', serve, {'document_root': settings.STATIC_ROOT, 'path': 'favicon.ico'}),
]

# Always add static files, not just in debug mode
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)