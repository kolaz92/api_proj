import torch
from torchvision.models import resnet18
import torchvision.transforms as T
import pickle
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def load_classes():
    '''
    Returns resnet classes
    '''
    with open('utils/classes_weather.pkl','rb') as f:
        labels = pickle.load(f)
    return list(labels.keys())

def class_id_to_label(i):
    '''
    Input int: class index
    Returns class name
    '''

    labels = load_classes()
    return labels[i]

class myRegNet(nn.Module):
    def __init__(self):
         super().__init__()
         self.model = resnet18(weights=None)
         self.model.fc = nn.Linear(512, 11)
         # замораживаем слои
         for i in self.model.parameters():
             i.requires_grad = False
        # размораживаем только последний, который будем обучать
         self.model.fc.weight.requires_grad = True
         self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)

def load_model(IsResNet: bool):
    if IsResNet:
        '''
        Returns resnet model with custom weights
        '''
        model = myRegNet()
        weights = 'utils/model_weights_weather.pth'
        model.load_state_dict(torch.load(weights,map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tmodel = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        return tmodel, tokenizer
    
def text2toxicity(text, tokenizer, model, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

def transform_image(img):
    '''
    Input: PIL img
    Returns: transformed image
    '''
    trnsfrms = T.Compose(
        [
            T.Resize((224, 224)), 
            T.ToTensor(),
            T.Normalize(mean, std)
        ]
    )
    return trnsfrms(img)

