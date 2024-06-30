#%%writefile test.py
from icevision.all import *
from icevision.models import *
from icevision.core import BaseRecord
import pandas as pd
from PIL import Image, ImageDraw
import glob
import matplotlib.pyplot as plt
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

model_loaded = model_from_checkpoint("/content/monkeycatdog_det.pth")
model_type = model_loaded["model_type"]
backbone = model_loaded["backbone"]
class_map = model_loaded["class_map"]
img_size = model_loaded["img_size"]

model_type.show_results(model_loaded["model"], val_set, detection_threshold=0.4)

infer_dl = model_type.infer_dl(val_set, batch_size=4, shuffle=False) #อนุมานให้้ valset ทำเป็น dataloader แล้วเอาไป ทำการสร้างbbox ผ่าน

#retinanet infer_dl = model_type.infer_dl(val_set, batch_size=4, shuffle=False): ทำการสร้าง DataLoader (infer_dl) จากชุดข้อมูลที่ต้องการทำนาย (val_set) โดยที่จำนวน batch คือ 4 และไม่มีการสลับข้อมูล (shuffle=False)
# โดย DataLoader นี้จะถูกใช้ในการส่งภาพไปให้กับโมเดล RetinaNet เพื่อทำการตรวจจับวัตถุ (object detection) บนภาพเหล่านั้นๆ และคืนค่า bounding box ที่โมเดลคาดการณ์ได้ในแต่ละภาพ.

preds = model_type.predict_from_dl(model_loaded["model"], infer_dl, keep_images=True) #นำ รูปใน bbox ไป classification ใน resnet 50

#preds = model_type.predict_from_dl(model_loaded["model"], infer_dl, keep_images=True): นำ DataLoader (infer_dl) ที่ได้มาใช้ในการทำนายด้วยโมเดลที่โหลดมาจาก checkpoint (model_loaded["model"])
# โดยทำการตรวจจับวัตถุในภาพที่อยู่ใน bounding box ที่ได้แล้ว และทำการจำแนกประเภทของวัตถุใน bounding box โดยใช้ ResNet-50 ที่เรียนรู้มาแล้วจากภาพที่อยู่ใน bounding box นั้นๆ ซึ่งผลลัพธ์ที่ได้จะเป็นรายการของ predictions พร้อมกับภาพที่ถูกจัดประเภทไว้ในแต่ละ bounding box.

#พารามิเตอร์ keep_images=True ที่ถูกใช้ในฟังก์ชัน model_type.predict_from_dl() มีหน้าที่บอกให้ฟังก์ชันเก็บภาพที่ผ่านการประมวลผล (inference) โดยโมเดลไว้ด้วย นั่นหมายความว่า เมื่อเราทำการทำนาย (inferencing) บนภาพใน DataLoader แล้ว
# ไม่เพียงแค่ส่งผลลัพธ์ที่เป็นการจัดประเภท (classification) หรือตรวจจับวัตถุ (object detection) กลับมาเท่านั้น แต่ยังเก็บภาพต้นฉบับที่ผ่านการประมวลผลของโมเดลไว้ด้วยด้วย
#การทำให้ฟังก์ชัน predict_from_dl() เก็บภาพต้นฉบับนี้เอาไว้เป็นไปได้เพื่อใช้ในการแสดงผลหรือการตรวจสอบผลลัพธ์ทีหลัง หรือในการวิเคราะห์เพิ่มเติมโดยไม่ต้องใช้ภาพต้นฉบับใหม่เสมอไป ซึ่งเป็นประโยชน์ในการลดการใช้ทรัพยากรและเพิ่มประสิทธิภาพของการทำงานโดยรวม

## สมมติว่า preds เป็นผลลัพธ์ที่ได้จาก predict_from_dl()
#for result in preds:
    # ดูคีย์ทั้งหมดใน dictionary ของผลลัพธ์
#    print(result.keys())

plt.figure(figsize=(12, 12))
imgA = show_preds(preds=preds[:4])
plt.axis('off')
plt.savefig("/content/imgRESULT1.jpg")
plt.close()

img = Image.open((glob.glob(os.path.join(r"/content/data/yolo-animal-detection-small/train","*.jpg")))[4])
pred_dict  = model_type.end2end_detect(img, valid_tramform, model_loaded["model"], class_map=class_map, detection_threshold=0.5)
imgg = pred_dict['img']
imgg.save("/content/imgRESULT2.jpg")

print(pred_dict.keys())

for i in pred_dict.keys():
  print(pred_dict[i])

