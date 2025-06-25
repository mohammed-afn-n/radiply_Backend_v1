# D:\Radiply Backend\fetchimage\views.py
from rest_framework import viewsets
from rest_framework import generics
from rest_framework import viewsets
import csv
from django.http import JsonResponse
from django.shortcuts import render
from django.http import  Http404
from django.http import FileResponse
from django.http import JsonResponse, Http404, HttpResponse
from fetchimage.models import DataSet, Pathologies
from fetchimage.serializers import DataSetSerializer,PathologiesSerializer
import radiply as radiply

from rest_framework import viewsets, status
from django.views.decorators.csrf import csrf_exempt
import json
import torch
import numpy as np
from fetchimage.models import Worklist, Pathologies,Heatmap
from .serializers import WorklistSerializer
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
from django.contrib.auth.hashers import check_password
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action
import os


d = radiply.datasets.NIH_Dataset(views=["PA", "AP", "L"], unique_patients=False)
def get_csv_data(request):
    file_path = "/mnt/efs/common/radiply/radiply_v0/combined_test_dataset_details.csv"
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



transforms = torchvision.transforms.Compose([radiply.datasets.XRayCenterCrop(), radiply.datasets.XRayResizer(224)])
def get_image(request, image_name):
    data_set_name = request.GET.get("dataSetName", None)
 

    if not data_set_name:
        return JsonResponse({"error": "dataSetName parameter is required"}, status=400)

    if data_set_name == "NIH":
        d = radiply.datasets.NIH_Dataset(views=["PA", "AP", "L"], unique_patients=False)
        image_index = d.csv[d.csv["Image Index"] == image_name].index[0]
        image_path = d.imgpath / d.csv["Image Index"].iloc[image_index]

    elif data_set_name == "CHEX":
        d = radiply.datasets.CheX_Dataset(transform=transforms, data_aug=True, unique_patients=False, views=["PA", "AP", "L"])
        sanitized_image_name = image_name.replace("!", "/")
       
        image_index = d.csv[d.csv["Path"] == sanitized_image_name].index[0]
        
        imgid = d.csv["Path"].iloc[image_index]
        imgid = imgid.replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0/", "")
        image_path = d.imgpath / imgid
        #image_path ="/mnt/efs/common/radiply/data/external/CheXpert-mirrored-Data/"+imgid
        print(f"d.imgpath {d.imgpath}")
        
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
        print(f"d.imgpath {d.imgpath}")

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
            
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            results = radiply.utils.process_images(image_paths, device)
            
            for result in results:
                for key, value in result.items():
                    if isinstance(value, np.float32):
                        result[key] = float(value)
                    elif isinstance(value, torch.Tensor):
                        result[key] = value.item()
                
                #selected_rows['img'] = result["filename"].split("/")[-1]
                
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
                            "data_set_name": selected_rows['dataSetName']
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

                heatmaps, original_img = radiply.utils.generate_heatmap3(image_paths)
               
            # Convert original image to bytes
                original_img_bytes = image_to_base64(original_img)

            # Create Heatmap object
                heatmap_instance = Heatmap.objects.create(
                    worklist=worklist,
                    original_img=original_img_bytes
                )

            # Save each pathology heatmap to the correct field
                for pathology, heatmap_img in heatmaps:
                    heatmap_img_bytes = image_to_base64(heatmap_img)  # Convert to bytes
                    field_name = pathology.lower().replace(" ", "_")  # Convert to field name format
                
                # Check if the field exists in the model
                    if hasattr(heatmap_instance, field_name):
                        setattr(heatmap_instance, field_name, heatmap_img_bytes)  # Set binary field value

            # Save the updated heatmap instance
                heatmap_instance.save()
                
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
       
        
        # Initialize a list to store the results
        combined_data = []

        # Loop through selected options and process each dataset
        for option in selected_options:
            # Determine the file path and dataset based on selected option
            if option == 'NIH':
                file_path = "/mnt/efs/common/radiply/radiply_v0/Radiply Backend/nih.csv"
                data_set_name = 'nih'
                fieldnames = ['Patient ID','Finding Labels', 'Image Index', 'View Position','Patient Age','Patient Gender']
            elif option == 'BBox_Pn_NIH':
                
                file_path = "/mnt/efs/common/radiply/radiply_v0/data/external/NIH-Data/BBox_Pn_NIH.csv"
                data_set_name = 'BBox_Pn_NIH'
                fieldnames = ['Finding Label', 'Image Index']
            elif option == 'CHEX':
                file_path = "/mnt/efs/common/radiply/radiply_v0/Radiply Backend/chex.csv"
                data_set_name = 'chex'
                fieldnames = ['patientid', 'Path', 'view','Age','Sex']
            elif option == 'SIIM':
                file_path = "/mnt/efs/common/radiply/radiply_v0/Radiply Backend/siim.csv"
                data_set_name = 'siim'
                fieldnames = ['patientid', 'ImageId', 'view',]
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
                                'patient_ID':row['Patient ID'],
                                'name': row['Finding Labels'],
                                'img': row['Image Index'],
                                'view': row['View Position'],
                                'patient_age':row['Patient Age'],
                                'patient_gender':row['Patient Gender'],
                                'dataSetName': 'NIH'
                            })
                        elif data_set_name == 'BBox_Pn_NIH':
                          
                            combined_data.append({
                                'patient_ID':'',
                                'name': row['Finding Label'],
                                'img': row['Image Index'],
                                'view': '',
                                'patient_age':'',
                                'patient_gender':'',
                                'dataSetName': 'BBox_Pn_NIH'
                            })
                        elif data_set_name == 'chex':
                            combined_data.append({
                                'patient_ID':row['patientid'],
                                'name': row['patientid'],
                                'img': row['Path'],
                                'view': row['view'],
                                'patient_age':row['Age'],
                                'patient_gender':row['Sex'],
                                'dataSetName': 'CHEX'
                            })
                        elif data_set_name == 'siim':
                            combined_data.append({
                                'patient_ID':row['patientid'],
                                'name': row['patientid'],
                                'img': row['ImageId'],
                                'view': '',
                                'patient_age':'',
                                'patient_gender':'',
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


@csrf_exempt
def generate_heatmap_view(request):
    if request.method == 'POST':
        try:
            # Get data from the request
            data = json.loads(request.body)
            worklist_id = data.get('id')
            label = data.get('label')
           

            # Get imgpath using worklist_id from your database (e.g., Worklist model)
            worklist = Worklist.objects.get(id=worklist_id)
            imgpath = worklist.imgpath
         

            # Generate heatmaps
            heatmaps, original_img = radiply.utils.generate_heatmap(imgpath, label)

            # Convert original image to bytes
            original_img_bytes = image_to_base64(original_img)

            # Create Heatmap object
            heatmap_instance = Heatmap.objects.create(
                worklist=worklist,
                original_img=original_img_bytes
            )

            # Save each pathology heatmap to the correct field
            for pathology, heatmap_img in heatmaps:
                heatmap_img_bytes = image_to_base64(heatmap_img)  # Convert to bytes
                field_name = pathology.lower().replace(" ", "_")  # Convert to field name format
                
                # Check if the field exists in the model
                if hasattr(heatmap_instance, field_name):
                    setattr(heatmap_instance, field_name, heatmap_img_bytes)  # Set binary field value

            # Save the updated heatmap instance
            heatmap_instance.save()

            return JsonResponse({"status": "success", "message": "Heatmaps saved successfully."})

        except Worklist.DoesNotExist:
            return JsonResponse({"status": "error", "message": "Worklist ID not found."}, status=404)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

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


