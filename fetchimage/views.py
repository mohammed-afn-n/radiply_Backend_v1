# D:\Radiply Backend\fetchimage\views.py
from rest_framework import viewsets
from .serializers import UserSerializer
from rest_framework import generics
from fetchimage.models import User
from rest_framework.response import Response
from rest_framework import viewsets
import csv
from django.http import JsonResponse
from django.shortcuts import render
from django.http import  Http404
from django.http import FileResponse
from fetchimage.models import DataSet, Pathologies
from fetchimage.serializers import DataSetSerializer,PathologiesSerializer
import radiply
from rest_framework import viewsets, status
from django.views.decorators.csrf import csrf_exempt
import json
import torch
import numpy as np
from fetchimage.models import Worklist, Pathologies
d = radiply.datasets.NIH_Dataset(views=["PA", "AP", "L"], unique_patients=False)
def get_csv_data(request):
    file_path = "/mnt/efs/common/radiply/combined_test_dataset_details.csv"
    data = []
    
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data.append({
                    'name': row['Finding Labels'],
                    'img': row['Image Index'],
                    'view':row['View Position']
                })
        return JsonResponse(data, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import (
    IsAuthenticated,
    IsAdminUser,
    AllowAny
)

from django.contrib.auth.hashers import check_password
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action

import os




IMAGE_DIR = r"C:/Users/moham/Downloads"

def get_image(request, image_name):
    print(f'my img : {image_name}')
    # Validate image_name
    image_index = d.csv[d.csv["Image Index"] == image_name].index[0]  # Get the index for the image name
    image_path = d.imgpath / d.csv["Image Index"].iloc[image_index]
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            return JsonResponse({"image": img.read().decode("latin1")})
    else:
        raise Http404("Image not found")



class DataSetPagination(PageNumberPagination):
    page_size = 10  # Set the default page size
    page_size_query_param = 'page_size'  # Allow the client to define page size (optional)
    max_page_size = 100  # Set a maximum page size for security reasons

class DataSetViewSet(viewsets.ModelViewSet):
    queryset = DataSet.objects.all()
    serializer_class = DataSetSerializer
    pagination_class = DataSetPagination  # Add the pagination class

class PathologiesViewSet(viewsets.ModelViewSet):
    queryset = Pathologies.objects.all()
    serializer_class = PathologiesSerializer

    # def get_queryset(self):
    #     queryset.filter()

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes=[IsAuthenticated]
    
    def create(self, request, *args, **kwargs):
        # Get data from the request
        data = request.data

        # Set default values for admin user creation
        data['is_staff'] = True  # Set the user as staff
        data['is_superuser'] = True  # Set the user as a superuser
        data['is_active'] = True 
        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)

        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(
        detail=False,
        methods=['GET'],
        url_path='login-user'
    )
    def get_user(self, request, *args, **kwargs):
        print("===============================")
        try:
            user = self.get_queryset().get(id=request.user.id)
            serializer = self.get_serializer(user)
            return Response(serializer.data)
        except User.DoesNotExist:
            return Response({"error": "User not found."}, status=404)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_password(request):
    user = request.user
    data = request.data

    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if not current_password or not new_password:
        return Response({"error": "Both current and new passwords are required."}, status=400)

    if not check_password(current_password, user.password):
        return Response({"error": "Current password is incorrect."}, status=400)

    user.set_password(new_password)
    user.save()
    return Response({"message": "Password updated successfully!"}, status=200)

@csrf_exempt
def send_selected_rows(request):
    if request.method == "POST":
        try:
            # Parse JSON payload
            data = json.loads(request.body)
            selected_rows = data.get("selectedRows", [])  # List of image names

            # List to store image paths
            image_paths = []

            # Iterate through selected rows to find the image paths
            for image_name in selected_rows:
                if image_name in d.csv["Image Index"].values:
                    image_index = d.csv[d.csv["Image Index"] == image_name].index[0]
                    image_path = d.imgpath / d.csv["Image Index"].iloc[image_index]
                    image_paths.append(image_path)
                else:
                    print(f"Image name {image_name} not found in the DataFrame.")

            print("Image paths to process:", image_paths)

            # Process the images using Radiply
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            results = radiply.utils.process_images(image_paths, device)

            # Convert results to JSON-serializable format (float32 to float)
            for result in results:
                for key, value in result.items():
                    if isinstance(value, np.float32):  # Check for np.float32
                        result[key] = float(value)  # Convert to Python float
                    elif isinstance(value, torch.Tensor):  # If value is a tensor
                        result[key] = value.item()  # Convert tensor to Python float

            # Store data in Worklist and Pathologies tables
            for result in results:
                # Extract the image name from the filename
                image_name = result["filename"].split("/")[-1]

                # Create or update Worklist entry
                worklist, created = Worklist.objects.get_or_create(
                    study_name=image_name,  # Use image name as study name
                    defaults={
                        "study_start_date": "2023-01-01",  # Replace with actual date
                        "study_end_date": "2023-01-01",  # Replace with actual date
                        "modality": "CT",  # Replace with actual modality
                        "status": "Pending",  # Replace with actual status
                        "priority": "High",  # Replace with actual priority
                        "imgpath": str(result["filename"]),  # Store the full image path
                        "img_name": image_name,  # Store the image name
                    }
                )

                # Create or update Pathologies entry
                pathologies, created = Pathologies.objects.get_or_create(
                    worklist=worklist,
                    defaults={
                        "atelectasis": result.get("Atelectasis", 0.0),
                        "cardiomegaly": result.get("Cardiomegaly", 0.0),
                        "consolidation": result.get("Consolidation", 0.0),
                        "edema": result.get("Edema", 0.0),
                        "effusion": result.get("Effusion", 0.0),
                        "emphysema": result.get("Emphysema", 0.0),
                        "fibrosis": result.get("Fibrosis", 0.0),
                        "hernia": result.get("Hernia", 0.0),
                        "infiltration": result.get("Infiltration", 0.0),
                        "mass": result.get("Mass", 0.0),
                        "nodule": result.get("Nodule", 0.0),
                        "pleural_thickening": result.get("Pleural_Thickening", 0.0),
                        "pneumonia": result.get("Pneumonia", 0.0),
                        "pneumothorax": result.get("Pneumothorax", 0.0),
                        "enlarged_cardiomediastinum": result.get("Enlarged Cardiomediastinum", 0.0),
                        "fracture": result.get("Fracture", 0.0),
                        "lung_lesion": result.get("Lung Lesion", 0.0),
                        "lung_opacity": result.get("Lung Opacity", 0.0),
                    }
                )

            # Prepare the response data
            response_data = {
                "success": True,
                "results": results,  # Ensure results are JSON-serializable
            }
            return JsonResponse(response_data, safe=False)

        except Exception as e:
            # Log the error and send a response
            print(f"Error occurred: {e}")
            response_data = {
                "success": False,
                "error": str(e),
            }
            return JsonResponse(response_data, status=500)