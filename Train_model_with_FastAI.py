#%%writefile train.py
import fastai.vision
import fastai.vision.all
from icevision.all import *
from icevision.models import *
import os 
from icevision.core import BaseRecord
import pandas as pd
import fastai

row_traincsv_data_path = os.path.join(r"/content/data/train.csv")
df = pd.read_csv(row_traincsv_data_path)

df.rename(columns={"class":"label"},inplace=True)

train_annotation_path = os.path.join(r"/content/data/yolo-animal-detection-small","train_annotation.csv")
df.to_csv(train_annotation_path,index=False)
df = pd.read_csv(train_annotation_path)

template_record = ObjectDetectionRecord()

class AmDetec(Parser):
    def __init__(self,template_record):
        super().__init__(template_record=template_record)

        self.df = pd.read_csv(train_annotation_path)
        self.class_map = ClassMap(list(self.df["label"].unique())) # {"background":0,"cat":1,"dog":2}

    def __iter__(self) -> Any:
        for o in self.df.itertuples():
            yield o   #จะคืนค่าทีละแถวแทนที่จะคืนค่าทั้งหมดในคราวเดียว

    def __len__(self) -> int:
        return len(self.df)
    
    def record_id(self,o) -> Hashable:
        return o.filename
    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(os.path.join(r"/content/data/yolo-animal-detection-small","train",o.filename)) #setไฟล์pathเพื่อ train 
            record.set_img_size(ImgSize(width=o.width,height=o.height))
            record.detection.set_class_map(self.class_map)
        record.detection.add_bboxes([BBox.from_xyxy(o.xmin,o.ymin,o.xmax,o.ymax)])
        record.detection.add_labels([o.label])

parser = AmDetec(template_record=template_record)
train_record,val_record = parser.parse()
model_type = models.mmdet.retinanet
backbone = model_type.backbones.resnet50_fpn_1x

train_tramform = tfms.A.Adapter([*tfms.A.aug_tfms(size=224,presize=512),tfms.A.Normalize()])
valid_tramform = tfms.A.Adapter([*tfms.A.resize_and_pad(224),tfms.A.Normalize()])


train_set = Dataset(train_record,train_tramform)
val_set = Dataset(val_record,valid_tramform)    

train_dl = model_type.train_dl(train_set,batch_size=8,num_workers=4,shuffle =True)
val_dl = model_type.train_dl(train_set,batch_size=8,num_workers=4,shuffle =False)



model = model_type.model(backbone=backbone(pretrained=True),num_classes=len(parser.class_map)) # model_type คือ object detection model ของเราชื่ออะไร ส่่วน classification เป็นตัส backbone


metrics = [COCOMetric(metric_type=COCOMetricType.bbox)] #mAP score

#ใช้ fast AI ในการ train model

csv_logger = fastai.vision.all.CSVLogger(fname='training_log.csv')

# เพิ่ม COCOMetric ใน callbacks เพื่อบันทึกค่า AP และ AR ในแต่ละ epoch




learner = model_type.fastai.learner(dls=[train_dl,val_dl],model=model,metrics=metrics,cbs=csv_logger)

#lr = learner.lr_find()
#print(lr)
#learner.fine_tune(5,1e-4,freeze_epochs=1)
learner.fine_tune(50, 7e-5, freeze_epochs=1)
m = ClassMap(list(df['label'].unique()))
m.get_classes()

checkpoint_path = "monkeycatdog_det.pth"
save_icevision_checkpoint(
    model, 
    model_name='mmdet.retinanet', 
    backbone_name='resnet50_fpn_1x',
    classes=m.get_classes(),
    img_size=224,
    filename=checkpoint_path,
    meta={"icevision_version": "0.12.0"}
)