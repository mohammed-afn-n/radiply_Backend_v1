from django.shortcuts import render
from .serializers import UserSerializer, UserListSerializer
from rest_framework.decorators import action
from rest_framework.permissions import (
    IsAuthenticated,
    IsAdminUser,
    AllowAny
)
from .models import User
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import viewsets

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]
    # filter_backends = [SearchFilter]
    # search_fields = ['username', 'email']

    def get_serializer_class(self):
        if self.action == 'list':
            return UserListSerializer
        return super().get_serializer_class()


    def get_queryset(self):
        return super().get_queryset()

    @action(
        detail=False,
        methods=['GET'],
        url_path='login-user'
    )
    def get_user(self, request, *args, **kwargs):
        
        try:
            user = self.get_queryset().get(id=request.user.id)
            serializer = self.get_serializer(user)
            return Response(serializer.data)
        except User.DoesNotExist:
            return Response({"error": "User not found."}, status=404)