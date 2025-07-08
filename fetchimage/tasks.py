import requests
from celery import shared_task
from .models import StudyInstance
from datetime import datetime


from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
@shared_task
def fetch_new_dicom_studies():
    try:
        studies_res = requests.get("http://3.111.210.119:8042/studies")
        studies_res.raise_for_status()
        study_ids = studies_res.json()

        for study_id in study_ids:
            # ✅ Skip if already exists
            if StudyInstance.objects.filter(study_id=study_id).exists():
                continue

            study_res = requests.get(f"http://3.111.210.119:8042/studies/{study_id}")
            study_res.raise_for_status()
            study = study_res.json()

            main_tags = study.get("MainDicomTags", {})
            patient_tags = study.get("PatientMainDicomTags", {})
            series_list = study.get("Series", [])

            for series_id in series_list:
                instance_res = requests.get(f"http://3.111.210.119:8042/series/{series_id}/instances")
                instance_res.raise_for_status()
                instance_list = instance_res.json()

                for instance in instance_list:
                    instance_id = instance.get("ID")

                    if StudyInstance.objects.filter(instance_id=instance_id).exists():
                        continue

                    tags_res = requests.get(
                        f"http://3.111.210.119:8042/instances/{instance_id}/tags?expand"
                    )
                    tags_res.raise_for_status()
                    tags_data = tags_res.json()

                    modality = tags_data.get("0008,0060", {}).get("Value", "")
                    view_position = tags_data.get("0018,5101", {}).get("Value", "")
                    body_part_examined = (
                        tags_data.get("0018,0015", {}).get("Value", "") or
                        tags_data.get("0018,1030", {}).get("Value", "")
                    )

                    # ✅ Save to DB
                    new_instance = StudyInstance.objects.create(
                        study_id=study.get("ID", ""),
                        series_id=series_id,
                        instance_id=instance_id,
                        accession_number=main_tags.get("AccessionNumber", ""),
                        study_date=main_tags.get("StudyDate", ""),
                        study_description=main_tags.get("StudyDescription", ""),
                        patient_id=patient_tags.get("PatientID", ""),
                        patient_name=patient_tags.get("PatientName", ""),
                        patient_sex=patient_tags.get("PatientSex", ""),
                        patient_birth_date=patient_tags.get("PatientBirthDate", ""),
                        modality=modality,
                        view_position=view_position,
                        gender=patient_tags.get("PatientSex", ""),
                        body_part_examined=body_part_examined,
                        age=calculate_age_from_string(patient_tags.get("PatientBirthDate", "")),
                        status="Unread"  # Default status (change if needed)
                    )

                    # ✅ Send WebSocket update
                    channel_layer = get_channel_layer()
                    async_to_sync(channel_layer.group_send)(
                          "group_row_updates",
                        {
                            "type": "row_update_message",
                            "message": {
                                "action": "new_row",
                                "data": {
                                    "id": new_instance.id,
                                    "study_id": new_instance.study_id,
                                    "series_id": new_instance.series_id,
                                    "instance_id": new_instance.instance_id,
                                    "accession_number": new_instance.accession_number,
                                    "study_date": new_instance.study_date,
                                    "study_description": new_instance.study_description,
                                    "patient_id": new_instance.patient_id,
                                    "patient_name": new_instance.patient_name,
                                    "patient_sex": new_instance.patient_sex,
                                    "patient_birth_date": new_instance.patient_birth_date,
                                    "modality": new_instance.modality,
                                    "view_position": new_instance.view_position,
                                    "gender": new_instance.gender,
                                    "body_part_examined": new_instance.body_part_examined,
                                    "age": new_instance.age,
                                    "status": new_instance.status,
                                }
                            }
                        }
                    
                    )

        print("✅ All new study instance data saved.")

    except requests.RequestException as e:
        print("❌ Error during fetch:", e)
        

def calculate_age_from_string(date_str):
    try:
        birth_date = datetime.strptime(date_str, "%Y%m%d")
        today = datetime.today()
        return today.year - birth_date.year - (
            (today.month, today.day) < (birth_date.month, birth_date.day)
        )
    except (ValueError, TypeError):
        return None
