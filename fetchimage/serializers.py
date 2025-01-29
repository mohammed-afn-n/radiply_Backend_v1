#D:\Radiply Backend\fetchimage\serializers.py
from rest_framework import serializers
# from django.contrib.auth.models import User
from fetchimage.models import User, DataSet,Pathologies


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        exclude = ['password']

class DataSetSerializer(serializers.ModelSerializer):  

    class Meta:
        model = DataSet
        fields = ['id', 'name']


class PathologiesSerializer(serializers.ModelSerializer): 

    class Meta:
        model = Pathologies
        fields = '__all__'