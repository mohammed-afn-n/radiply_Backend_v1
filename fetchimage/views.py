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
from django.http import JsonResponse, Http404, HttpResponse
from fetchimage.models import DataSet, Pathologies
from fetchimage.serializers import DataSetSerializer,PathologiesSerializer
import radiply
from rest_framework import viewsets, status
from django.views.decorators.csrf import csrf_exempt
import json
import torch
import numpy as np
from fetchimage.models import Worklist, Pathologies
from .serializers import WorklistSerializer
import torchvision
from PIL import Image
import pydicom
import os
import io
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import base64
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
                    'view':row['View Position'],
                    'dataSetName': 'NIH'
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
transforms = torchvision.transforms.Compose([radiply.datasets.XRayCenterCrop(), radiply.datasets.XRayResizer(224)])
def get_image(request, image_name):
    data_set_name = request.GET.get("dataSetName", None)
    print(f"Dataset Name: {data_set_name}")
    print(f"My Image: {image_name}")

    if not data_set_name:
        return JsonResponse({"error": "dataSetName parameter is required"}, status=400)

    if data_set_name == "NIH":
        d = radiply.datasets.NIH_Dataset(views=["PA", "AP", "L"], unique_patients=False)
        image_index = d.csv[d.csv["Image Index"] == image_name].index[0]
        image_path = d.imgpath / d.csv["Image Index"].iloc[image_index]

    elif data_set_name == "CHEX":
        d = radiply.datasets.CheX_Dataset(transform=transforms, data_aug=True, unique_patients=False, views=["PA", "AP", "L"])
        sanitized_image_name = image_name.replace("!", "/")
        print(f'sanitized_image_name {sanitized_image_name}')
        image_index = d.csv[d.csv["Path"] == sanitized_image_name].index[0]
        print(f'image_index {image_index}')
        imgid = d.csv["Path"].iloc[image_index]
        imgid = imgid.replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0/", "")
        image_path = d.imgpath / imgid
        print(f'image_path : {image_path}')
        
    elif data_set_name == "SIIM":
        d = radiply.datasets.SIIM_Pneumothorax_Dataset(transform=transforms, data_aug=True, unique_patients=False)
        image_id = image_name
        image_index = d.csv[d.csv["ImageId"] == image_id].index[0]
        imgid = d.csv['ImageId'].iloc[image_index]
        img_path = d.file_map[imgid + ".dcm"]
        
        # Read the DICOM image
        dicom_img = pydicom.dcmread(img_path)
        img_array = dicom_img.pixel_array  # Get NumPy array

        # Convert to 8-bit grayscale
        img_array = (img_array / img_array.max() * 255).astype(np.uint8)

        # Convert NumPy array to PIL image
        img = Image.fromarray(img_array)

        # Save image to buffer
        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        img_io.seek(0)

        # Read image data and convert to 'latin1' encoded string
        img_data = img_io.read().decode("latin1")

        return JsonResponse({"image": img_data})

    else:
        return JsonResponse({"error": "Invalid dataSetName"}, status=400)

    # For NIH and CHEX, check if the image exists and return it
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

def process_image_paths(selected_rows):
    image_paths = []
    image_name = selected_rows.get("img")
    dataset_name = selected_rows.get("dataSetName")
    
    if dataset_name == "NIH":
        d = radiply.datasets.NIH_Dataset(views=["PA", "AP", "L"], unique_patients=False)
        image_index = d.csv[d.csv["Image Index"] == image_name].index[0]
        image_path = d.imgpath / d.csv["Image Index"].iloc[image_index]
        image_paths.append(image_path)
    if dataset_name == "BBox_Pn_NIH":
        d = radiply.datasets.NIH_Dataset(views=["PA", "AP", "L"], unique_patients=False)
        image_index = d.csv[d.csv["Image Index"] == image_name].index[0]
        image_path = d.imgpath / d.csv["Image Index"].iloc[image_index]
        image_paths.append(image_path)
    
    elif dataset_name == "CHEX":
        d = radiply.datasets.CheX_Dataset(transform=None, data_aug=True, unique_patients=False, views=["PA", "AP", "L"])
        sanitized_image_name = image_name.replace("!", "/")
        image_index = d.csv[d.csv["Path"] == sanitized_image_name].index[0]
        imgid = d.csv["Path"].iloc[image_index].replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0/", "")
        image_path = d.imgpath / imgid
        image_paths.append(image_path)
    
    elif dataset_name == "SIIM":
        d = radiply.datasets.SIIM_Pneumothorax_Dataset(transform=None, data_aug=True, unique_patients=False)
        image_index = d.csv[d.csv["ImageId"] == image_name].index[0]
        imgid = d.csv['ImageId'].iloc[image_index]
        image_path = d.file_map[imgid + ".dcm"]
        image_paths.append(image_path)
    
    return image_paths


@csrf_exempt
def send_selected_rows(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            selected_rows = data.get("selectedRow", {})
            image_paths = process_image_paths(selected_rows)
            print("Image paths to process:", image_paths)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            results = radiply.utils.process_images(image_paths, device)
            
            for result in results:
                for key, value in result.items():
                    if isinstance(value, np.float32):
                        result[key] = float(value)
                    elif isinstance(value, torch.Tensor):
                        result[key] = value.item()
                
                selected_rows['img'] = result["filename"].split("/")[-1]
                
                # Check if the record already exists in Worklist
                worklist_exists = Worklist.objects.filter(img_name=selected_rows['img']).exists()
                
                if not worklist_exists:
                    worklist, _ = Worklist.objects.get_or_create(
                        study_name=selected_rows['img'],  
                        defaults={
                            "study_start_date": "2023-01-01",  
                            "study_end_date": "2023-01-01",  
                            "modality": "Xray", 
                            "status": "Pending",  
                            "priority": "High",  
                            "imgpath": str(result["filename"]),  
                            "img_name": selected_rows['img'],
                            "data_set_name": selected_rows['img']
                        }
                    )
                    
                    Pathologies.objects.get_or_create(
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
                else:
                    print(f"Skipping {selected_rows['img']} as it already exists in Worklist.")

                # Broadcast the updated row to all clients
                channel_layer = get_channel_layer()
                async_to_sync(channel_layer.group_send)(
                    "group_row_updates",
                    {
                        'type': 'row_update_message',
                        'message': {
                            'action': 'update_row',
                            'row_id': selected_rows['img'],
                            'new_data': {
                                'status': 'Completed'
                            }
                        }
                    }
                )

            return JsonResponse({"success": True, "results": results}, safe=False)
        
        except Exception as e:
            print(f"Error occurred: {e}")
            return JsonResponse({"success": False, "error": str(e)}, status=500)        

@csrf_exempt
def send_selected_options(request):
    if request.method == "POST":
        data = json.loads(request.body)
        selected_options = data.get("selectedOptions", [])

        # Print the selected options for debugging
        print("Selected Options:", selected_options)
        
        # Initialize a list to store the results
        combined_data = []

        # Loop through selected options and process each dataset
        for option in selected_options:
            # Determine the file path and dataset based on selected option
            if option == 'NIH':
                file_path = "/mnt/efs/common/radiply/Radiply Backend/nih.csv"
                data_set_name = 'nih'
                fieldnames = ['Finding Labels', 'Image Index', 'View Position']
            elif option == 'BBox_Pn_NIH':
                print("i have entered")
                file_path = "/mnt/efs/common/radiply/data/external/NIH-Data/BBox_Pn_NIH.csv"
                data_set_name = 'BBox_Pn_NIH'
                fieldnames = ['Finding Label', 'Image Index']
            elif option == 'CHEX':
                file_path = "/mnt/efs/common/radiply/Radiply Backend/chex.csv"
                data_set_name = 'chex'
                fieldnames = ['patientid', 'Path', 'view']
            elif option == 'SIIM':
                file_path = "/mnt/efs/common/radiply/Radiply Backend/siim.csv"
                data_set_name = 'siim'
                fieldnames = ['patientid', 'ImageId', 'view']
            else:
                continue  # Skip invalid options

            # Read the CSV file and process rows based on dataset
            try:
                with open(file_path, mode='r', encoding='utf-8') as file:
                    csv_reader = csv.DictReader(file)
                    for row in csv_reader:
                        # Dynamically handle the dataset
                        if data_set_name == 'nih':
                            combined_data.append({
                                'name': row['Finding Labels'],
                                'img': row['Image Index'],
                                'view': row['View Position'],
                                'dataSetName': 'NIH'
                            })
                        elif data_set_name == 'BBox_Pn_NIH':
                            print("i have entered2")
                            combined_data.append({
                                'name': row['Finding Label'],
                                'img': row['Image Index'],
                                'view': '',
                                'dataSetName': 'BBox_Pn_NIH'
                            })
                        elif data_set_name == 'chex':
                            combined_data.append({
                                'name': row['patientid'],
                                'img': row['Path'],
                                'view': row['view'],
                                'dataSetName': 'CHEX'
                            })
                        elif data_set_name == 'siim':
                            combined_data.append({
                                'name': row['patientid'],
                                'img': row['ImageId'],
                                'view': '',
                                'dataSetName': 'SIIM'
                            })

            except Exception as e:
                return JsonResponse({"error": str(e)}, status=500)

        # Return combined data for all selected datasets
        return JsonResponse(combined_data, safe=False)
    
    
@api_view(['GET'])
def get_all_studies(request):
    studies = Worklist.objects.all()
    serializer = WorklistSerializer(studies, many=True)
    return Response(serializer.data)


@csrf_exempt
def get_worklist_id(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            img_name = data.get("img_name")

            worklist = Worklist.objects.filter(img_name=img_name).first()
            if worklist:
                return JsonResponse({"id": worklist.id}, status=200)
            else:
                return JsonResponse({"error": "Worklist not found"}, status=404)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def get_pathologies(request, worklist_id):
    try:
        pathologies = Pathologies.objects.filter(worklist_id=worklist_id).values()
        if pathologies.exists():
            return JsonResponse(list(pathologies), safe=False, status=200)
        return JsonResponse({"error": "No pathologies found"}, status=404)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)




@csrf_exempt
def generate_heatmap_view(request):
    if request.method == 'POST':
        try:
            # Get data from the request
            data = json.loads(request.body)
            worklist_id = data.get('id')
            label = data.get('label')
            print(f'worklist_id : {worklist_id}   :  label :{label}')

            # Get imgpath using worklist_id from your database (e.g., Worklist model)
            worklist = Worklist.objects.get(id=worklist_id)
            imgpath = worklist.imgpath
            print(f'imgpath : {imgpath}   :  label :{label}')

            # Assuming 'generate_heatmap' returns a heatmap image and the original image
            heatmap_img, original_img =radiply.utils.generate_heatmapDens(imgpath, label)
            print(f'heatmap_img : {heatmap_img}   :  original_img :{original_img}')
            # Convert images to base64-encoded strings
            heatmap_img_base64 = image_to_base64(heatmap_img)
            original_img_base64 = image_to_base64(original_img)

            return JsonResponse({
                'success': True,
                'heatmapImg': heatmap_img_base64,
                'originalImg': original_img_base64
            })

        except Exception as e:
            print(f"Error: {e}")
            return JsonResponse({'success': False, 'message': str(e)})

def image_to_base64(image_array): 
    # Convert the image to a PIL image, then to base64
    image = Image.fromarray(image_array.astype(np.uint8))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('latin1')
    return img_str




import traceback
import pandas as pd

@csrf_exempt
def get_csvreport_data(request):
    CSV_FILE_PATH = "/mnt/efs/common/radiply/data/external/NIH-Data/BBox_Pn_NIH.csv"
    
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            report_name = data.get("reportName")
            print(f'report_name: {report_name}')

            if not report_name:
                return JsonResponse({"error": "Missing reportName"}, status=400)

            # Read CSV file
            if not os.path.exists(CSV_FILE_PATH):
                return JsonResponse({"error": "CSV file not found"}, status=500)

            df = pd.read_csv(CSV_FILE_PATH)
            print(f'CSV Loaded Successfully, Columns: {df.columns.tolist()}')

            # Ensure 'Image Index' column exists
            if "Image Index" not in df.columns:
                return JsonResponse({"error": "'Image Index' column not found in CSV"}, status=500)

            # Find row where 'Image Index' matches the reportName
            row = df[df["Image Index"].astype(str).str.strip() == str(report_name).strip()]
            print(f'row: {row}')

            if row.empty:
                return JsonResponse({"error": "Report not found"}, status=404)

            # Convert row to dictionary and return as JSON
            row_dict = row.to_dict(orient="records")[0]
            print(f'row_dict: {row_dict}')

            return JsonResponse({"success": True, "data": row_dict})

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())  # Log full traceback
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)




