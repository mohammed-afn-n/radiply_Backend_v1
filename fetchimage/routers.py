# D:\Radiply Backend\fetchimage\routers.py
from rest_framework.routers import DefaultRouter
from fetchimage.views import DataSetViewSet,PathologiesViewSet
api_router = DefaultRouter()
api_router.register('datasets', DataSetViewSet)
api_router.register('pathologies', PathologiesViewSet)   

# api_router.register('worklist',StudiesViewset)