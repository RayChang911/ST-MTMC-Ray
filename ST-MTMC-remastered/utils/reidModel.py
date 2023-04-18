import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import torch.optim as optim
from keras.applications.resnet import preprocess_input
from keras.preprocessing import image
from keras.engine.training import Model
from keras.saving.save import load_model
from keras.applications.resnet import ResNet50

import random
import cv2
import csv
import numpy as np
import logging
from PIL import Image
from copy import deepcopy

from utils.utils import loadReidModel

logging.basicConfig(level=logging.INFO)
"""
class ReidModel(nn.Module):
    def __init__(self, class_num = 751) -> None:
        super().__init__()
        model = models.resnet50(pretrained=True)
        self.model = model
        classifier = nn.Sequential(*[nn.Linear(2048, class_num)])
        self.classifier = classifier

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

class featureExtractor(ReidModel):
    def __init__(self, class_num=751) -> None:
        super().__init__(class_num)
        self.load_state_dict(torch.load('./utils/resnet50-market1501.pt'))
        self.createInference()

    def createInference(self):
        selfcopy = deepcopy(self)
        features = list(selfcopy.children())[:-1]
        features[-1] = nn.Sequential(*list(features[-1].children())[:-1])
        self = nn.Sequential(*features)

def inference(inferenceModel: featureExtractor, crop = np.ndarray) -> torch.Tensor:
    crop = np.moveaxis(crop, 2, 0)
    INPUT_HW = (256, 128)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inferenceModel.to(device)
    t = torch.from_numpy(crop).float()

    test_transform = transforms.Compose([
        transforms.Resize(INPUT_HW, interpolation=3),
    ])
    with torch.no_grad():
        t:torch.Tensor = test_transform(t)
        t = t.to(device)
        t = torch.unsqueeze(t, dim = 0)
        t = inferenceModel(t)
        t = torch.squeeze(t)
    return t

def compareTwo(tensors1: list[torch.Tensor], tensors2: list[torch.Tensor]) -> float:
    # cos = nn.CosineSimilarity(dim=0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        sims = torch.tensor([]).to(device)
        for i in range(len(tensors1)):
            tensor1 = tensors1[i]
            tensor1_norm = tensor1 / tensor1.norm(dim=0)
            tensor1_norm = torch.unsqueeze(tensor1_norm, dim = 0)
            cal_mat = torch.stack(tensors2)
            cal_mat_norm = cal_mat / cal_mat.norm(dim = 1)[:, None]
            res = torch.mm(tensor1_norm, cal_mat_norm.transpose(0, 1))
            res = torch.squeeze(res, 0)
            sims = torch.cat((sims, res), dim = 0)
    ret =  torch.mean(sims).item()
    return ret
"""
class ReidModel():
    def __init__(self):
        self.model_resnet_person = loadReidModel('./utils/model/model_resnet50_person.h5')
        self.model_resnet_car = loadReidModel('./utils/model/model_resnet50_car.h5')
        self.model_resnet_motorcycle = loadReidModel('./utils/model/model_resnet50_motorcycle.h5')
        # pretrain
        model_resnet_pretrain = ResNet50(weights = 'imagenet')
        model_resnet_pretrain =  Model(inputs=model_resnet_pretrain.input, outputs=model_resnet_pretrain.layers[
        len(model_resnet_pretrain.layers) - 2].output)
        self.model_resnet_pretrain = model_resnet_pretrain
        logging.info('Load ResNet50 Pretrain Model')
        # classifier
        resnet_classifier = load_model('./utils/model/model_resnet50_classifier_V3.h5')
        resnet_classifier = Model(inputs=resnet_classifier.input, outputs=resnet_classifier.output)
        self.resnet_classifier = resnet_classifier
        logging.info('Load ResNet50 Classifier Model')


    def createInference_reid(self, cls):
        if cls == "person":
            return self.model_resnet_person
        elif cls == "car":
            return self.model_resnet_car
        elif cls == "motorcycle":
            return self.model_resnet_motorcycle
        else:
            return self.model_resnet_pretrain
    def createInference_classifer(self):
        return self.resnet_classifier

def inference(crop, cls, ReidModel) -> list:

    reidModel = ReidModel.createInference_reid(cls)

    img = cv2.resize(crop, (244, 244))
    if cls == "undefined":
        img = cv2.resize(crop, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    MOG_res_feature = reidModel.predict(img)[0]
    MOG_res_feature = [float(round(x, 3)) for x in MOG_res_feature]
    
    return MOG_res_feature


def compareTwo(tensor1: list, tensor2: list) -> float:
    ret = round(np.dot(tensor1,tensor2)/(np.linalg.norm(tensor1) * np.linalg.norm(tensor2)), 3)

    return ret
