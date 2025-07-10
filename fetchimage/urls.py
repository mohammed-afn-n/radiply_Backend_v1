
#D:\Radiply Backend\fetchimage\urls.py
from django.urls import path, include
from . import views
from fetchimage.routers import api_router, study_router
from rest_framework.routers import DefaultRouter
from .views import StudyInstanceViewSet


urlpatterns = [
   
   
    path('api/update-password/', views.update_password, name='update_password'),
    path('send-selected/', views.send_selected_rows, name='send_selected'),
   
    path('worklist/', views.get_all_studies, name='get_all_studies'),
    path('get_worklist_id/', views.get_worklist_id, name='get_worklist_id'),
    path('get_pathologies/<int:worklist_id>/', views.get_pathologies, name='get_pathologies'),
  
    path("get_csvreport_data/", views.get_csvreport_data, name="get_csvreport_data"),
    
    
    path('', include(study_router.urls)),
    path('image/', views.get_image_from_orthanc, name='get_image_from_orthanc'),
    path('get_report/', views.get_report, name='get_report'),
    path('upload/', views.upload_dicom, name='upload_dicom'),
    path('orthanc/studies/<str:study_id>/series/', views.get_study_series, name='get_study_series'),
    path('orthanc/series/<str:series_id>/instances/', views.get_series_instances, name='get_series_instances'),
]

urlpatterns += api_router.urls
