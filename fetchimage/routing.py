# routing.py
from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path("ws/row_updates/", consumers.RowUpdateConsumer.as_asgi()),
]