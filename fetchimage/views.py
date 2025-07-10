# D:\Radiply Backend\fetchimage\views.py
from rest_framework import viewsets,filters
from rest_framework import generics

import csv
from django.http import JsonResponse
from django.shortcuts import render
from django.http import  Http404
from django.http import FileResponse
from django.http import JsonResponse, Http404, HttpResponse
from fetchimage.models import DataSet, Pathologies
from fetchimage.serializers import DataSetSerializer,PathologiesSerializer

import requests
from rest_framework import viewsets, status
from django.views.decorators.csrf import csrf_exempt
import json
import torch
import numpy as np
from fetchimage.models import Worklist, Pathologies,Heatmap,StudyInstance,AIResult
from .serializers import WorklistSerializer,StudyInstanceSerializer
import torchvision
from PIL import Image
import pydicom
import os
import io
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import base64
from django.utils import timezone
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import (
    IsAuthenticated,
    IsAdminUser,
    AllowAny
)
from django.contrib.auth.hashers import check_password, make_password

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action
import os
from io import BytesIO
import torchxrayvision as xrv
from django_filters.rest_framework import DjangoFilterBackend






# class DataSetPagination(PageNumberPagination):
#     page_size = 10  # Set the default page size
#     page_size_query_param = 'page_size'  # Allow the client to define page size (optional)
#     max_page_size = 100  # Set a maximum page size for security reasons

class DataSetViewSet(viewsets.ModelViewSet):
    queryset = DataSet.objects.all()
    serializer_class = DataSetSerializer
    # pagination_class = DataSetPagination  # Add the pagination class

class PathologiesViewSet(viewsets.ModelViewSet):
    queryset = Pathologies.objects.all()
    serializer_class = PathologiesSerializer

    # def get_queryset(self):
    #     queryset.filter()

  
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.contrib.auth import get_user_model

User = get_user_model()

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_password(request):
    data = request.data

    user_id = data.get('user_id')
    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if not user_id or not current_password or not new_password:
        return Response({"error": "User ID, current password, and new password are required."}, status=400)

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return Response({"error": "User not found."}, status=404)

    # Only allow the user to update their own password, or add extra logic for admins
    if request.user.id != user.id and not request.user.is_superuser:
        return Response({"error": "You are not authorized to update this user's password."}, status=403)

    if not user.check_password(current_password):
        return Response({"error": "Current password is incorrect."}, status=400)

    user.set_password(new_password)
    user.save()

    return Response({"message": "Password updated successfully!"}, status=200)




@csrf_exempt
def send_selected_rows(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            selected_rows = data.get("selectedRow", {})
            instance_id = selected_rows.get('instance_id')
            if not instance_id:
                return JsonResponse({"error": "Missing instance_id"}, status=400)

            # Step 1: Fetch DICOM from Orthanc
            orthanc_url = f"http://3.111.210.119:8042/instances/{instance_id}/file"
            orthanc_response = requests.get(orthanc_url)
            if orthanc_response.status_code != 200:
                return JsonResponse({"error": "Failed to fetch from Orthanc"}, status=500)

            dicom_bytes = BytesIO(orthanc_response.content)
            dicom_data = pydicom.dcmread(dicom_bytes)

            if not hasattr(dicom_data, "PixelData"):
                return JsonResponse({"error": "No image data in DICOM"}, status=400)

            pixel_array1 = dicom_data.pixel_array.astype(np.float32)
            # pixel_array = dicom_data.pixel_array

            # # Normalize and convert to uint8 for image preview
            # if pixel_array.dtype != np.uint8:
            #     pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.ptp() or 1) * 255
            #     pixel_array = pixel_array.astype(np.uint8)

            # if len(pixel_array.shape) == 2:
            #     image = Image.fromarray(pixel_array)
            # elif len(pixel_array.shape) == 3:
            #     image = Image.fromarray(pixel_array, "RGB")
            # else:
            #     return JsonResponse({"error": "Unsupported image format"}, status=400)

            # buffered = BytesIO()
            # image.save(buffered, format="JPEG")
            # image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Step 2: Check if AIResult already exists
            study_instance = StudyInstance.objects.filter(instance_id=instance_id).first()
            if not study_instance:
                return JsonResponse({"error": "StudyInstance not found for instance_id."}, status=404)

            existing_result = AIResult.objects.filter(study_instance=study_instance).first()

            if existing_result:
                # If result exists, return it directly
                results = [{
                    field.name: getattr(existing_result, field.name)
                    for field in AIResult._meta.fields
                    if field.name not in ['id', 'study_instance', 'created_at']
                }]
            else:
                # Else, process using AI
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                results =process_images_ai1(pixel_array1, device)

                if not results:
                    return JsonResponse({"error": "AI processing failed or returned no data."}, status=500)

                # Convert values to Python-native types
                for result in results:
                    for key, value in result.items():
                        if isinstance(value, (np.float32, np.float64)):
                            result[key] = float(value)
                        elif isinstance(value, torch.Tensor):
                            result[key] = value.item()

                print(results, 'results')
                ai_output = results[0]  # Only save the first result
                print(ai_output, 'ai_output')

                AIResult.objects.create(
                    study_instance=study_instance,
                    atelectasis=ai_output.get("Atelectasis", 0.0),
                    cardiomegaly=ai_output.get("Cardiomegaly", 0.0),
                    consolidation=ai_output.get("Consolidation", 0.0),
                    edema=ai_output.get("Edema", 0.0),
                    effusion=ai_output.get("Effusion", 0.0),
                    emphysema=ai_output.get("Emphysema", 0.0),
                    fibrosis=ai_output.get("Fibrosis", 0.0),
                    hernia=ai_output.get("Hernia", 0.0),
                    infiltration=ai_output.get("Infiltration", 0.0),
                    mass=ai_output.get("Mass", 0.0),
                    nodule=ai_output.get("Nodule", 0.0),
                    pleural_thickening=ai_output.get("Pleural_Thickening", 0.0),
                    pneumonia=ai_output.get("Pneumonia", 0.0),
                    pneumothorax=ai_output.get("Pneumothorax", 0.0),
                    enlarged_cardiomediastinum=ai_output.get("Enlarged Cardiomediastinum", 0.0),
                    fracture=ai_output.get("Fracture", 0.0),
                    lung_lesion=ai_output.get("Lung Lesion", 0.0),
                    lung_opacity=ai_output.get("Lung Opacity", 0.0),
                )

                study_instance.status = "Finalized"
                study_instance.save()

            # Step 3: Broadcast update
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                "group_row_updates",
                {
                    'type': 'row_update_message',
                    'message': {
                        'action': 'update_row',
                        'row_id': selected_rows.get('id'),
                        'new_data': {
                            'status': 'Finalized'
                        }
                    }
                }
            )

            return JsonResponse({
                "success": True
            }, safe=False)

        except Exception as e:
            print(f"Error occurred: {e}")
            return JsonResponse({"success": False, "error": str(e)}, status=500)
default_pathologies = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]
def process_images_ai1(image, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    outputs = []

    # Normalize image
    img = xrv.utils.normalize(image, image.max())

    # Ensure image is 2D (remove channel dimension if present)
    if len(img.shape) > 2:
        img = img[:, :, 0]

    # Add channel dimension (C x H x W)
    img = img[None, :, :]

    # Apply transform: center crop and resize
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)

    # Convert to torch tensor and add batch dimension (B x C x H x W)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # Load and prepare model
    model = xrv.models.get_model('densenet121-res224-all')
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        preds = model(img_tensor)
        probs = preds[0] * 100  # convert to percentage

    # Get index of the pathology
    pathologies =default_pathologies
    output = {pathology: probs[i].item() for i, pathology in enumerate(pathologies)}
    outputs.append(output)
   

    return outputs
    
    
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

def image_to_base64Send(binary_data):
    """Convert binary image data to a base64 string."""
    if binary_data:
        return base64.b64encode(binary_data).decode('utf-8')
    return None

@csrf_exempt
def get_pathologies(request, worklist_id):
    try:
        # Fetch pathologies
        pathologies = list(Pathologies.objects.filter(worklist_id=worklist_id).values())

        # Fetch heatmap data
        heatmap = Heatmap.objects.filter(worklist_id=worklist_id).first()

        if heatmap:
            heatmap_data = {
                "original_img": image_to_base64Send(heatmap.original_img),
                "pneumothorax": image_to_base64Send(heatmap.pneumothorax),
                "consolidation": image_to_base64Send(heatmap.consolidation),
                "enlarged_cardiomediastinum": image_to_base64Send(heatmap.enlarged_cardiomediastinum),
                "lung_lesion": image_to_base64Send(heatmap.lung_lesion),
                "pneumonia": image_to_base64Send(heatmap.pneumonia),
                "infiltration": image_to_base64Send(heatmap.infiltration),
                "effusion": image_to_base64Send(heatmap.effusion),
                "atelectasis": image_to_base64Send(heatmap.atelectasis),
                "cardiomegaly": image_to_base64Send(heatmap.cardiomegaly),
                "edema": image_to_base64Send(heatmap.edema),
                "lung_opacity": image_to_base64Send(heatmap.lung_opacity),
                "fracture": image_to_base64Send(heatmap.fracture),
                "mass": image_to_base64Send(heatmap.mass),
                "nodule": image_to_base64Send(heatmap.nodule),
                "emphysema": image_to_base64Send(heatmap.emphysema),
                "fibrosis": image_to_base64Send(heatmap.fibrosis),
                "pleural_thickening": image_to_base64Send(heatmap.pleural_thickening),
                "hernia": image_to_base64Send(heatmap.hernia),
            }
        else:
            heatmap_data = None  # No heatmap found

        return JsonResponse({"pathologies": pathologies, "heatmap": heatmap_data}, safe=False, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



def image_to_base64(image):
    """Convert image (NumPy array) to base64-encoded bytes."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    image = Image.fromarray(image.astype(np.uint8))  # Convert to PIL image
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    
    return buffered.getvalue()  




import traceback
import pandas as pd

@csrf_exempt
def get_csvreport_data(request):
    CSV_FILE_PATH = "/mnt/efs/common/radiply/radiply_v0/data/external/NIH-Data/BBox_Pn_NIH.csv"
    
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            report_name = data.get("reportName")
           

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
            

            if row.empty:
                return JsonResponse({"error": "Report not found"}, status=404)

            # Convert row to dictionary and return as JSON
            row_dict = row.to_dict(orient="records")[0]
           

            return JsonResponse({"success": True, "data": row_dict})

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())  # Log full traceback
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)



#new 


class CustomPageNumberPagination(PageNumberPagination):
    page_size = 15
    page_size_query_param = 'page_size'

class StudyInstanceViewSet(viewsets.ModelViewSet):
    queryset = StudyInstance.objects.all().order_by('-study_date')
    serializer_class = StudyInstanceSerializer
    pagination_class = CustomPageNumberPagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["study_id", "series_id", "instance_id", "accession_number", "study_date", "study_description", "patient_id", "patient_name", "patient_sex", "patient_birth_date", "modality", "view_position", "body_part_examined", "gender", "created_at", "status", "received_at", "age"]
    search_fields = ['patient_id', 'patient_name', 'accession_number', 'study_description']
    
    
@csrf_exempt
def get_image_from_orthanc(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            instance_id = data.get("instance_id")

            if not instance_id:
                return HttpResponseBadRequest("Missing instance_id")

            # Step 1: Get raw DICOM file from Orthanc
            orthanc_url = f"http://3.111.210.119:8042/instances/{instance_id}/file"
            orthanc_response = requests.get(orthanc_url)

            if orthanc_response.status_code != 200:
                return JsonResponse({"error": "Failed to fetch from Orthanc"}, status=500)

            # Step 2: Convert DICOM to JPEG
            dicom_bytes = BytesIO(orthanc_response.content)
            dicom_data = pydicom.dcmread(dicom_bytes)

            # Check if pixel data exists
            if not hasattr(dicom_data, "PixelData"):
                return JsonResponse({"error": "No image data in DICOM"}, status=400)

            # Get pixel array and convert to uint8
            pixel_array = dicom_data.pixel_array

            # Normalize pixel data to 0-255
            if pixel_array.dtype != np.uint8:
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.ptp() or 1) * 255
                pixel_array = pixel_array.astype(np.uint8)

            # Handle grayscale or RGB
            if len(pixel_array.shape) == 2:  # Grayscale
                image = Image.fromarray(pixel_array)
            elif len(pixel_array.shape) == 3:  # RGB
                image = Image.fromarray(pixel_array, "RGB")
            else:
                return JsonResponse({"error": "Unsupported image format"}, status=400)

            # Step 3: Save JPEG
            # Step 3: Convert image to base64 directly in memory
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Step 4: Return base64 to frontend
            return JsonResponse({
                "image_data": image_base64
            })


        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return HttpResponseBadRequest("Invalid request method")


@csrf_exempt
def get_report(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            selected_rows = data.get("selectedRow", {})
            instance_id = selected_rows.get('instance_id')
            if not instance_id:
                return JsonResponse({"error": "Missing instance_id"}, status=400)

            # Step 1: Fetch DICOM from Orthanc
            orthanc_url = f"http://3.111.210.119:8042/instances/{instance_id}/file"
            orthanc_response = requests.get(orthanc_url)
            if orthanc_response.status_code != 200:
                return JsonResponse({"error": "Failed to fetch from Orthanc"}, status=500)

            dicom_bytes = BytesIO(orthanc_response.content)
            dicom_data = pydicom.dcmread(dicom_bytes)

            if not hasattr(dicom_data, "PixelData"):
                return JsonResponse({"error": "No image data in DICOM"}, status=400)

            pixel_array1 = dicom_data.pixel_array.astype(np.float32)
            pixel_array = dicom_data.pixel_array

            # Normalize and convert to uint8 for image preview
            if pixel_array.dtype != np.uint8:
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.ptp() or 1) * 255
                pixel_array = pixel_array.astype(np.uint8)

            if len(pixel_array.shape) == 2:
                image = Image.fromarray(pixel_array)
            elif len(pixel_array.shape) == 3:
                image = Image.fromarray(pixel_array, "RGB")
            else:
                return JsonResponse({"error": "Unsupported image format"}, status=400)

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Step 2: Check if AIResult already exists
            study_instance = StudyInstance.objects.filter(instance_id=instance_id).first()
            if not study_instance:
                return JsonResponse({"error": "StudyInstance not found for instance_id."}, status=404)

            existing_result = AIResult.objects.filter(study_instance=study_instance).first()
            results=[{}]
            if existing_result:
                # If result exists, return it directly
                results = [{
                    field.name: getattr(existing_result, field.name)
                    for field in AIResult._meta.fields
                    if field.name not in ['id', 'study_instance', 'created_at']
                }]
            

            return JsonResponse({
                "success": True,
                "results": results,
                "image_base64": image_base64
            }, safe=False)

        except Exception as e:
            print(f"Error occurred: {e}")
            return JsonResponse({"success": False, "error": str(e)}, status=500)
        
ORTHANC_URL = 'http://3.111.210.119:8042/instances'     
@csrf_exempt
def upload_dicom(request):
    if request.method == 'POST':
        print('hi')
        files = request.FILES.getlist('dicom_files')
        responses = []

        for file in files:
            response = requests.post(
                ORTHANC_URL,
                headers={'Content-Type': 'application/dicom'},
                data=file.read()
            )
            responses.append(response.json())

        return JsonResponse({'status': 'success', 'details': responses})

    return JsonResponse({'error': 'Invalid request'}, status=400)


ORTHANC_URL1 = 'http://3.111.210.119:8042'

@api_view(['GET'])
def get_study_series(request, study_id):
    """Get all series for a study"""
    try:
        response = requests.get(f"{ORTHANC_URL1}/studies/{study_id}/series")
        response.raise_for_status()
        return JsonResponse(response.json(), safe=False)
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['GET'])
def get_series_instances(request, series_id):
    """Get all instances for a series"""
    try:
        response = requests.get(f"{ORTHANC_URL1}/series/{series_id}/instances")
        response.raise_for_status()
        return JsonResponse(response.json(), safe=False)
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)