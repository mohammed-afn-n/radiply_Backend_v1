
#D:\Radiply Backend\fetchimage\urls.py
from django.urls import path
from . import views
from fetchimage.routers import api_router

urlpatterns = [
    path('get-csv-data/', views.get_csv_data, name='get_csv_data'),
    path("image/<str:image_name>/", views.get_image, name="get_image"),
    path('api/update-password/', views.update_password, name='update_password'),
    path('send-selected/', views.send_selected_rows, name='send_selected'),
    path('sendcsv-selected-options/', views.send_selected_options, name='send_selected_options'),
    path('worklist/', views.get_all_studies, name='get_all_studies'),
    path('get_worklist_id/', views.get_worklist_id, name='get_worklist_id'),
    path('get_pathologies/<int:worklist_id>/', views.get_pathologies, name='get_pathologies'),
    path('generate_heatmap/', views.generate_heatmap_view, name='generate_heatmap'),
    path("get_csvreport_data/", views.get_csvreport_data, name="get_csvreport_data"),
   
]

urlpatterns += api_router.urls
