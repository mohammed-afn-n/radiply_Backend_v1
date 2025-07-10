#/mnt/efs/common/radiply/Worklist/radiplyBackend/app/serializers.py
from rest_framework import serializers
from .models import User
from django.contrib.auth.hashers import check_password

class UserListSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__' 

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'
        extra_kwargs = {
            'password': {'write_only': True, 'required': False}  
        }

    def create(self, validated_data):
        """
        Create a new user instance. Password is required during creation.
        """
        password = validated_data.pop('password', None) 
        if not password:
            raise serializers.ValidationError({"password": "This field is required."})
        
        # Create the user instance
        instance = self.Meta.model(**validated_data)
        
        # Set password and save
        instance.set_password(password)
        instance.save()
        
        # Refresh the instance from database and verify
        instance.refresh_from_db()
        if not check_password(password, instance.password):
            raise serializers.ValidationError({"password": "Password was not properly hashed."})
        print('done')
        return instance

    def update(self, instance, validated_data):
        """
        Update an existing user instance. Password is optional during updates.
        """
        print('i have entered ')
        password = validated_data.pop('password', None)  
        m2m_fields = {}
        for attr, value in list(validated_data.items()):
            if isinstance(self.fields[attr], serializers.ManyRelatedField):
                m2m_fields[attr] = value
                validated_data.pop(attr)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password is not None and password.strip() != "": 
            instance.set_password(password)
        instance.save()
        for attr, value in m2m_fields.items():
            getattr(instance, attr).set(value)

        return instance
    
    
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.conf import settings

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)
        # Add custom claims
        data['alert_before_expiration'] = settings.SIMPLE_JWT['ALERT_BEFORE_EXPIRATION'].total_seconds() * 1000
        # Include refresh token in response
        data['refresh'] = data.pop('refresh', None)
        return data

from rest_framework_simplejwt.serializers import TokenRefreshSerializer
import logging

logger = logging.getLogger(__name__)

class CustomTokenRefreshSerializer(TokenRefreshSerializer):
    def validate(self, attrs):
        logger.debug(f"Refresh Token Received: {attrs.get('refresh')}")
        data = super().validate(attrs)
        data['alert_before_expiration'] = settings.SIMPLE_JWT['ALERT_BEFORE_EXPIRATION'].total_seconds() * 1000
        return data
