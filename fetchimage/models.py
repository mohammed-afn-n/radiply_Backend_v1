#D:\Radiply Backend\fetchimage\models.py
from django.db import models

class DataSet(models.Model):
    name =  models.CharField(max_length=200)
    description = models.TextField()
 
class Worklist(models.Model):
    study_name = models.CharField(max_length=255, verbose_name="Study Name")
    study_start_date = models.DateField(verbose_name="Study Start Date")
    study_end_date = models.DateField(verbose_name="Study End Date")
    modality = models.CharField(max_length=50)
    status = models.CharField(max_length=50)
    priority = models.CharField(max_length=50)
    imgpath = models.CharField(max_length=255, verbose_name="Image Path")
    img_name = models.CharField(max_length=255, verbose_name="Image Name")
    data_set_name = models.CharField(max_length=255, verbose_name="Data Set Name")  

    class Meta:
        verbose_name = "Worklist"
        verbose_name_plural = "Worklists"

    def __str__(self):
        return self.study_name


class Pathologies(models.Model):
    worklist = models.OneToOneField(Worklist, on_delete=models.CASCADE, related_name='pathology')

    atelectasis = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    cardiomegaly = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    consolidation = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    edema = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    effusion = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    emphysema = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    fibrosis = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    hernia = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    infiltration = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    mass = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    nodule = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    pleural_thickening = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    pneumonia = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    pneumothorax = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    enlarged_cardiomediastinum = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    fracture = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    lung_lesion = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)
    lung_opacity = models.DecimalField(max_digits=15, decimal_places=5, default=0.00)

    class Meta:
        verbose_name = "Pathology"
        verbose_name_plural = "Pathologies"

    def __str__(self):
        return f"Pathology for {self.worklist.study_name if self.worklist else 'Unknown Study'}"




from django.utils import timezone
class Heatmap(models.Model):
    worklist = models.ForeignKey(Worklist, on_delete=models.CASCADE, related_name='heatmaps')
    generated_at = models.DateTimeField(default=timezone.now)

    # Column for the original image
    original_img = models.BinaryField(null=True, blank=True)

    # Columns for each pathology
    pneumothorax = models.BinaryField(null=True, blank=True)
    consolidation = models.BinaryField(null=True, blank=True)
    enlarged_cardiomediastinum = models.BinaryField(null=True, blank=True)
    lung_lesion = models.BinaryField(null=True, blank=True)
    pneumonia = models.BinaryField(null=True, blank=True)
    infiltration = models.BinaryField(null=True, blank=True)
    effusion = models.BinaryField(null=True, blank=True)
    atelectasis = models.BinaryField(null=True, blank=True)
    cardiomegaly = models.BinaryField(null=True, blank=True)
    edema = models.BinaryField(null=True, blank=True)
    lung_opacity = models.BinaryField(null=True, blank=True)
    fracture = models.BinaryField(null=True, blank=True)
    mass = models.BinaryField(null=True, blank=True)
    nodule = models.BinaryField(null=True, blank=True)
    emphysema = models.BinaryField(null=True, blank=True)
    fibrosis = models.BinaryField(null=True, blank=True)
    pleural_thickening = models.BinaryField(null=True, blank=True)
    hernia = models.BinaryField(null=True, blank=True)

    def __str__(self):
        return f"{self.worklist.id} - {self.generated_at}"
    
    
class StudyInstance(models.Model):
    id = models.AutoField(primary_key=True)
    study_id = models.CharField(max_length=255)
    series_id = models.CharField(max_length=255)
    instance_id = models.CharField(max_length=255)
    accession_number = models.CharField(max_length=255, blank=True)
    study_date = models.CharField(max_length=20, blank=True)
    study_description = models.TextField(blank=True)
    patient_id = models.CharField(max_length=255, blank=True)
    patient_name = models.CharField(max_length=255, blank=True)
    patient_sex = models.CharField(max_length=10, blank=True)
    patient_birth_date = models.CharField(max_length=20, blank=True)
    modality = models.CharField(max_length=50, blank=True)
    view_position = models .CharField(max_length=50, blank=True)
    body_part_examined = models.CharField(max_length=100, blank=True)
    gender = models.CharField(max_length=10, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, default='Unread') # or PROCESSING, COMPLETED
    received_at = models.DateTimeField(null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)

class AIResult(models.Model):
    id = models.AutoField(primary_key=True)
    study_instance = models.ForeignKey(StudyInstance, on_delete=models.CASCADE, related_name="results")
    created_at = models.DateTimeField(auto_now_add=True)

    # Example AI output fields
    atelectasis = models.FloatField(default=0.0)
    cardiomegaly = models.FloatField(default=0.0)
    consolidation = models.FloatField(default=0.0)
    edema = models.FloatField(default=0.0)
    effusion = models.FloatField(default=0.0)
    emphysema = models.FloatField(default=0.0)
    fibrosis = models.FloatField(default=0.0)
    hernia = models.FloatField(default=0.0)
    infiltration = models.FloatField(default=0.0)
    mass = models.FloatField(default=0.0)
    nodule = models.FloatField(default=0.0)
    pleural_thickening = models.FloatField(default=0.0)
    pneumonia = models.FloatField(default=0.0)
    pneumothorax = models.FloatField(default=0.0)
    enlarged_cardiomediastinum = models.FloatField(default=0.0)
    fracture = models.FloatField(default=0.0)
    lung_lesion = models.FloatField(default=0.0)
    lung_opacity = models.FloatField(default=0.0)
