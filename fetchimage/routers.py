# D:\Radiply Backend\fetchimage\routers.py
from rest_framework.routers import DefaultRouter
from fetchimage.views import DataSetViewSet,PathologiesViewSet, StudyInstanceViewSet

api_router = DefaultRouter()
api_router.register('datasets', DataSetViewSet)
api_router.register('pathologies', PathologiesViewSet)   

# api_router.register('worklist',StudiesViewset)

study_router = DefaultRouter()
study_router.register(r'studies', StudyInstanceViewSet, basename='studyinstance')


# from django.urls import re_path
# from fetchimage.consumers import StudyConsumer  # Make sure this import matches the path

# websocket_urlpatterns = [
#     re_path(r'ws/studies/$', StudyConsumer.as_asgi()),
# ]

