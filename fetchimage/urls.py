
#D:\Radiply Backend\fetchimage\urls.py
from django.urls import path
from . import views
from fetchimage.routers import api_router

urlpatterns = [
    path('get-csv-data/', views.get_csv_data, name='get_csv_data'),
    path("image/<str:image_name>/", views.get_image, name="get_image"),
    path('api/update-password/', views.update_password, name='update_password'),
    path('send-selected/', views.send_selected_rows, name='send_selected')
]

urlpatterns += api_router.urls
