#/mnt/efs/common/radiply/Worklist/radiplyBackend/app/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser


class Role(models.TextChoices):
    ADMINISTRATOR = 'ADMIN', 'Administrator'
    RADIOLOGIST = 'RAD', 'Radiologist'
    TECHNICIAN = 'TECH', 'Technician'
    FRONTDESK = 'FRONT', 'Front Desk'
    REFERRING_PHYSICIAN = 'PHYS', 'Referring Physician'

    

class User(AbstractUser):
    role = models.CharField(
        max_length=5,
        choices=Role.choices,
        default=Role.FRONTDESK,
        verbose_name='User Role'
    )
