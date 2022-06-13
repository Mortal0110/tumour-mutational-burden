from __future__ import print_function, division

import torch
from torchvision import transforms
import os
from torch.nn import functional as F
from utils.folder2lmdb import ImageFolderLMDB
import yaml
import tqdm
from utils.ImageFolderPaths import ImageFolderWithPaths
import matplotlib.pyplot as plt
import csv

with open(os.path.join(os.path.abspath('.'), 'config/config.yml'), 'r', encoding='utf8') as fs:
    cfg = yaml.load(fs, Loader=yaml.FullLoader)
data_dir = cfg['tiles_dir']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = torch.load(cfg['model_path'])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if cfg['use_lmdb'] is True:
    testset = ImageFolderLMDB(os.path.join(data_dir, 'test.lmdb'),
                          data_transforms['test'])
else:
    testset = ImageFolderWithPaths(os.path.join(data_dir, 'test'),
                                                      data_transforms['test'])

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=4)

person_prob_dict = {}

csvfile = open('details.csv', 'w', newline='')
writer = csv.writer(csvfile)
writer.writerow(['names', 'labels', 'predicted', 'prob_0', 'prob_1'])

with torch.no_grad():
    for data in tqdm.tqdm(testloader):
        images, labels, names = data
        outputs = model_ft(images.to(device))
        probability = F.softmax(outputs, dim=1).data.squeeze()
        _, predicted = torch.max(outputs.data, 1)
        probs = probability.cpu().numpy()

        idx = predicted.data.cpu().numpy()
        for i in range(labels.size(0)):
            file = '-'.join(names[i].split('-')[:3])
            if file not in person_prob_dict.keys():
                person_prob_dict[file] = {'prob_0': 0, 'prob_1': 0,
                                          'label': labels[i],'prob': 0,'predict': 0}
            if probs.ndim == 2:
                person_prob_dict[file]['prob_0'] += probs[i, 0]
                person_prob_dict[file]['prob_1'] += probs[i, 1]
            else:
                person_prob_dict[file]['prob_0'] += probs[0]
                person_prob_dict[file]['prob_1'] += probs[1]

        predicted = predicted.tolist()
        for i in range(labels.size(0)):
            probs0 = 0
            probs1 = 0
            if probs.ndim == 2:
                probs0 += probs[i][0]
                probs1 += probs[i][1]
            else:
                probs0 += probs[0]
                probs1 += probs[1]
            data0 = [names[i], labels[i].tolist(), predicted[i], probs0, probs1]
            writer.writerow(data0)

csvfile.close()
total = len(person_prob_dict)
correct = 0
for key in person_prob_dict.keys():
    person_prob_dict[key]['prob'] = person_prob_dict[key]['prob_1']  / (person_prob_dict[key]['prob_0'] + person_prob_dict[key]['prob_1'])
    predict = 0
    if person_prob_dict[key]['prob_0'] < person_prob_dict[key]['prob_1']:
        predict = 1
    if person_prob_dict[key]['label'] == predict:
        correct += 1
print(correct, total)
print('Accuracy of the network on test images: %d %%' % (
    100 * correct / total))
y_label=[]
y_pre=[]
for key in person_prob_dict.keys():
    y_label.append(person_prob_dict[key]['label'].numpy().tolist())
    y_pre.append(person_prob_dict[key]['prob'])

from sklearn.metrics import roc_curve
fpr, tpr, thersholds = roc_curve(y_label, y_pre)

from sklearn.metrics import auc
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

plt.xlim([-0.05, 1.05])  # ÉèÖÃx¡¢yÖáµÄÉÏÏÂÏÞ£¬ÒÔÃâºÍ±ßÔµÖØºÏ£¬¸üºÃµÄ¹Û²ìÍ¼ÏñµÄÕûÌå
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # ¿ÉÒÔÊ¹ÓÃÖÐÎÄ£¬µ«ÐèÒªµ¼ÈëÒ»Ð©¿â¼´×ÖÌå
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
plt.savefig("roc_auc.png")
print('auc:',roc_auc)

for key in person_prob_dict.keys():
    person_prob_dict[key]['label'] = person_prob_dict[key]['label'].tolist()

import json
json_dict = json.dumps(person_prob_dict)
dict_ = json.loads(json_dict)
with open("person_prob_dict.json", "w", encoding='utf-8') as f:
  # json.dump(person_prob_dict, f)
  json.dump(person_prob_dict, f, indent=2, sort_keys=True, ensure_ascii=False)
