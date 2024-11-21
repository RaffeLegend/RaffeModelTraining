import torch
import copy
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from engine.base_trainer import BaseModel

class Validator(BaseModel):
    def name(self):
        return 'Validator'

    def __init__(self, opt):
        super(Validator, self).__init__(opt)
        self.opt = opt  
        self.model = None
        self.model_dict = None
        self.loader = None

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def update(self, model, loader):
        self.model = model
        self.loader = loader

    @staticmethod
    def calculate_acc(y_true, y_pred, thres):
        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
        acc = accuracy_score(y_true, y_pred > thres)
        return r_acc, f_acc, acc 

    @staticmethod
    def find_best_threshold(y_true, y_pred):
        "We assume first half is real 0, and the second half is fake 1"

        N = y_true.shape[0]

        if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
            return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

        best_acc = 0 
        best_thres = 0 
        for thres in y_pred:
            temp = copy.deepcopy(y_pred)
            temp[temp>=thres] = 1 
            temp[temp<thres] = 0 

            acc = (temp == y_true).sum() / N  
            if acc >= best_acc:
                best_thres = thres
                best_acc = acc 
        
        return best_thres

    def forward(self):
        with torch.no_grad():
            y_true, y_pred = [], []
            print ("Length of validating dataset: %d" %(len(self.loader)))
            for img, label in self.loader:
                in_tens = img.cuda()

                y_pred.extend(torch.sigmoid(self.model(in_tens)).flatten().tolist())
                y_true.extend(label.flatten().tolist())

        self.y_true, self.y_pred = np.array(y_true), np.array(y_pred)

    def evaluate(self):
        best_thres = self.find_best_threshold(self.y_true, self.y_pred)
        self.ap = average_precision_score(self.y_true, self.y_pred)
        self.r_acc0, self.f_acc0, self.acc0 = self.calculate_acc(self.y_true, self.y_pred, 0.5)
        self.r_acc1, self.f_acc1, self.acc1 = self.calculate_acc(self.y_true, self.y_pred, best_thres)
        return self.ap, self.r_acc0, self.f_acc0, self.acc0