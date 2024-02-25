import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
from skimage import io, transform
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms, utils, models
from sklearn.preprocessing import label_binarize

PATH = 'PD_Label_Final_VGG16_17.csv'

df = pd.read_csv(PATH)
df.head()
df['person_id'] = df['Video Name'].apply(lambda x: x.split('_')[0])
person_id = df['person_id'].values.tolist()

transform = {'train': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),

             'test': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])}

df

#Custom Dataset
class DD_Dataset(Dataset):
    # instance attributes
    def __init__(self, df, csv_file, root_dir, transform=None):
        self.data = df
        self.root_dir = root_dir
        #change to number of columns in csv
        self.labels = np.asarray(self.data.iloc[:, 1])
        self.transform = transform

    # length of the dataset passed to this class method
    def __len__(self):
        return len(self.data)

    # get the specific image and labels given the index
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0]+'.jpg')
        image = Image.open(img_name)
        image = image.convert('RGB')
        image_label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, image_label

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders):
# def train_model(model, criterion, optimizer, num_epochs):
    #copy the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch, num_epochs-1))
        print("="*10)

        for phase in ['train']:
        # for phase in ['train', 'val']:
            if phase == 'train':
                model.train()

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            scheduler.step()    #original not commented

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(dataloaders):
    running_correct = 0
    running_total = 0
    true_labels = []
    pred_labels = []
    # no gradient calculation
    with torch.no_grad():
        for data in dataloaders['test']:
            # print('data', data)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print('inputs', inputs)
            # print(type(labels))
            print(labels)
            true_labels.append(labels.item())
            outputs = model_pre(inputs)
            _, preds = torch.max(outputs.data, 1)
            pred_labels.append(preds.item())
            running_total += labels.size(0)
            running_correct += (preds == labels).sum().item()
    return (true_labels, pred_labels, running_correct, running_total)

dataset = DD_Dataset(df=df,
                     csv_file=PATH,
                     root_dir='Downloads/PD_GEI_summary_79/')

def multiclassauc(y_true, y_score, n_classes=3):

    y_true_binary = label_binarize(y_true, classes=[0, 1, 2])
    auc_scores = []
    for i in range(n_classes):

        y_true_i = y_true_binary[:, i]
        y_score_i = [1 if pred == i else 0 for pred in y_score]


        auc_i = roc_auc_score(y_true_i, y_score_i)
        auc_scores.append(auc_i)

    mean_auc = sum(auc_scores) / len(auc_scores)
    return auc_scores, mean_auc

df['person_id'].unique()

df.info()

# k fold
k_fold = 5
random.seed(128)
unique_values = np.unique([int(i) for i in person_id])  # Retrieve unique values from the data
print('unique_values 1',unique_values)
unique_values = [i for i in unique_values if i != '70']
print('unique_values 2 ', unique_values)
unique_values = sorted(unique_values)
print('unique_values 3',unique_values)
random.shuffle(unique_values)  # Randomly shuffle unique values
unique_values = [str(i) for i in unique_values]
print('unique_values 4', unique_values)

total_result = []

fold_size = len(unique_values) // k_fold  # Calculate the size of each fold.
folds = [unique_values[i * fold_size:(i + 1) * fold_size] if i != (k_fold-1) else unique_values[i * fold_size:] for i in range(k_fold)]  
for k in range(k_fold):

    print('k_fold', k_fold)
    print('unique_values', unique_values)
    print('unique_values length', len(unique_values))
    test_indices = df[df['person_id'].isin(folds[k])].index.values.tolist()
    print('test_indices', test_indices)
    print('test_indices length', len(test_indices))

    test_person_id = df.loc[df['person_id'].isin(folds[k]), 'person_id'].values.tolist()
    print('test_person_id', test_person_id)
    print('test_person_id length', len(test_person_id))

    test_file_name = df.loc[df['person_id'].isin(folds[k]), 'Video Name'].values.tolist()
    train_indices = [i for i in df.index if i not in test_indices]
    dataset_sizes = {'train': len(train_indices),  'test': len(test_indices)}
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataset = DD_Dataset(df=df,
                               csv_file=PATH,
                               root_dir='Downloads/PD_GEI_summary_79/',
                               transform=transform['train'])
    test_dataset = DD_Dataset(df=df,
                              csv_file=PATH,
                              root_dir='Downloads/PD_GEI_summary_79/',
                              transform=transform['test'])

    # Check for cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=train_sampler),
                   'test': torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler)}
    model_pre = models.vgg16()
    model_pre.load_state_dict(torch.load("Downloads/pre_trained_models_ethan/vgg16-397923af.pth"))
    # don't calculate gradient since we will use the weights of pretrained model
    for param in model_pre.features.parameters():
        param.required_grad = False
    num_features = model_pre.classifier[6].in_features
    features = list(model_pre.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 3)])
    model_pre.classifier = nn.Sequential(*features)
    model_pre = model_pre.to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = torch.optim.SGD(model_pre.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # original not commmented
    EPOCHS = 20
    model_pre = train_model(model_pre,
                            criterion,
                            optimizer,
                            exp_lr_scheduler,
                            num_epochs=EPOCHS,
                            dataloaders=dataloaders)

    true_labels, pred_labels, running_correct, running_total = test_model(dataloaders=dataloaders)
    test_df = pd.DataFrame(np.array([test_person_id,test_file_name,pred_labels,true_labels]).T,
                               columns=['person_id', 'file_name', 'pred_label', 'true_label'])
    person_result = test_df.groupby('person_id').agg({'pred_label': pd.Series.mode,
                                                      'true_label': pd.Series.mode})
    person_result['pred_label'] = person_result['pred_label'].apply(lambda x: x if type(x) == str else x[-1])
    person_result['true_label'] = person_result['true_label'].apply(lambda x: x if type(x) == str else x[-1])
    # print(f"Fold [{k}/{k_fold}] Accuracy (person): {accuracy_score(person_result['true_label'], person_result['pred_label'],digits=4)}")
    print(f"Fold [{k}/{k_fold}] Accuracy (person): {accuracy_score(person_result['true_label'], person_result['pred_label']):.4f}")


    if len(total_result) == 0:
        total_result = person_result
    else:
        total_result = pd.concat([total_result, person_result])

    print(f'{k} the results of k-fold cross-validation for each classification')
    print(classification_report(person_result['true_label'],
                                person_result['pred_label'],
                                target_names=["Normal", "PD-Mild", "PD-Moderate"]))

    # Print the confusion matrix for the overall results
    conf_matrix_overall = confusion_matrix(total_result['true_label'], total_result['pred_label'])
    print("\nConfusion Matrix (Overall):")
    print(conf_matrix_overall)

    # print(f"Classification and overall AUC value are{multiclassauc(true_labels, pred_labels, n_classes=3)}")
    torch.save(model_pre.state_dict(), f'Downloads/pd_vgg16_79_fold_{k}.pth')


print('The final results of 5-fold cross-validation for each classification.')
print(classification_report(total_result['true_label'], total_result['pred_label'], target_names=["Normal", "PD-Mild", "PD-Moderate"],digits=4))
print('Overall prediction results')

# Convert predicted labels to strings
total_result['pred_label'] = total_result['pred_label'].astype(str)

# Verify the data types after conversion
print("Data types after conversion:")
print("True Labels:", total_result['true_label'].dtype)
print("Predicted Labels:", total_result['pred_label'].dtype)

# Print the classification report
print("\nClassification Report:")
print(classification_report(total_result['true_label'], total_result['pred_label'], target_names=["Normal", "PD-Mild", "PD-Moderate"], digits=4))

import pandas as pd

pd.set_option('display.max_rows', 100)  # Set the maximum number of rows to display

print(total_result)

# Convert predicted labels to strings
total_result['pred_label'] = total_result['pred_label'].astype(str)

# Verify the data types after conversion
print("Data types after conversion:")
print("True Labels:", total_result['true_label'].dtype)
print("Predicted Labels:", total_result['pred_label'].dtype)

# Print the classification report
print("\nClassification Report:")
print(classification_report(total_result['true_label'], total_result['pred_label'], target_names=["Normal", "PD-Mild", "PD-Moderate"], digits=4))

import pandas as pd

pd.set_option('display.max_rows', 100)  # Set the maximum number of rows to display

print(total_result)

# Assuming 'model_pre' is the trained VGG16 model
torch.save(model_pre.state_dict(), 'Downloads/pd_vgg16_79_model.pth')