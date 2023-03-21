from imageai.Detection.Custom import CustomObjectDetection
from imageai.Detection.Custom import DetectionModelTrainer
import os
import cv2


# 1. Train a pre-trained YOLOV3 model with our Data-Set
def train_pretrained_model(dataset_dir):
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=dataset_dir)
    trainer.setTrainConfig(object_names_array=["Timebox"], batch_size=4, num_experiments=10,
                           train_from_pretrained_model="pretrained-yolov3.h5")
    trainer.trainModel()


# 2. Evaluate the models generated by training
def evaluate_models(dataset_dir):
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=dataset_dir)
    trainer.evaluateModel(model_path="Dataset folder/models", json_path="Dataset folder/json/detection_config.json",
                          iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)


# 3. Detection of object with the best model
def detect_custom_object(in_frame):

    my_dir = r"E:\Career files\Degree Thesis\2. Dataset\Images Dataset\Object_Det_files/"
    os.chdir(my_dir)

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("detection_model-ex-013--loss-0003.658.h5")
    detector.setJsonPath("json/detection_config.json")
    detector.loadModel()
    detections, extracted_images = detector.detectObjectsFromImage(input_image=in_frame,
                                                                   output_image_path= "dump\detected.jpg",
                                                                   input_type="array", # ANALOGA TI THELW EDW
                                                                   extract_detected_objects=True)
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

    flag = False
    if extracted_images:
        img_det = extracted_images[0]
        flag = True
        return img_det, flag
    else:
        print("Didnt find any objects")
        return None, flag


    #img_d = cv2.imread(img_det)
    #print(type(img_d))
    #cv2.imshow("",img_d)
    #cv2.waitKey(0)

