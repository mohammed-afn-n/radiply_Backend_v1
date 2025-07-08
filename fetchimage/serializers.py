#D:\Radiply Backend\fetchimage\serializers.py
from rest_framework import serializers
# from django.contrib.auth.models import User
from fetchimage.models import DataSet,Pathologies,Worklist,StudyInstance



class StudyInstanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = StudyInstance
        fields = '__all__'

class DataSetSerializer(serializers.ModelSerializer):  

    class Meta:
        model = DataSet
        fields = ['id', 'name']


class PathologiesSerializer(serializers.ModelSerializer): 

    class Meta:
        model = Pathologies
        fields = '__all__'
        
from .models import Worklist

class WorklistSerializer(serializers.ModelSerializer):
    class Meta:
        model = Worklist
        fields = '__all__'
        
        
        
# serializers.py
