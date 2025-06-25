from rest_framework.routers import DefaultRouter
from .views import UserViewSet

app_router = DefaultRouter()
app_router.register('users', UserViewSet) 