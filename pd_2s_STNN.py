import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
import random
import re
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

random.seed(64)

data2 = np.load('pd_3d.npy', allow_pickle=True)
data3 = np.load('sil.npy')

person_id = data2[:, 0]
file_name_list = data2[:, 0]
person_id = [re.split("[_-]", i)[0] for i in person_id]
data2 = data2[:, 1:].astype(float)
combined_data = np.hstack((data3, data2))

print(combined_data.shape)

labels = combined_data[:, -1].astype(int)
print('labels', labels)
combined_data = combined_data[:, :-1]
print(combined_data.shape)

# Convert labels to one-hot encoding
num_classes = 3
one_hot_labels = np.eye(num_classes)[labels]

# Divide into 5-fold cross-validation
num_folds = 5
fold_size = int(combined_data.shape[0] / num_folds)

accuracies = []

def multiclassauc(y_true, y_score, n_classes=3):
    # Convert y_true into a binary matrix, where each column corresponds to a category.
    y_true_binary = label_binarize(y_true, classes=[0, 1, 2])
    auc_scores = []
    for i in range(n_classes):
        # Set the current class as positive and the other classes as negative
        y_true_i = y_true_binary[:, i]
        y_score_i = [1 if pred == i else 0 for pred in y_score]

        # Calculate the AUC value for the current class
        auc_i = roc_auc_score(y_true_i, y_score_i)
        auc_scores.append(auc_i)
    # Calculate the average AUC value
    mean_auc = sum(auc_scores) / len(auc_scores)
    return auc_scores, mean_auc

def split_data_kfold(person_id, file_name, data, label, k):
    unique_values = np.unique([int(i) for i in person_id])  # Retrieve unique values from the data
    unique_values = sorted(unique_values)
    random.shuffle(unique_values)  # Randomly shuffle unique values.
    unique_values = [str(i) for i in unique_values]

    fold_size = len(unique_values) // k  # Calculate the size of each fold
    folds = [unique_values[i * fold_size:(i + 1) * fold_size] if i != (k-1) else unique_values[i * fold_size:] for i in range(k)]  # Split unique values into k folds

    train_data = []
    test_data = []
    person_id_list = []
    file_list = []

    train_label = []
    test_label = []

    for i in range(k):
        test_ids = folds[i]  # The current fold as the test set
        train_ids = [id for id in unique_values if id not in test_ids]  # The remaining values as the training set

        # Extract file_path corresponding to the person_id in the training set.

        x = np.empty((0, data.shape[1]))
        for id, df in zip(person_id, data):
            if id in train_ids:
                x = np.concatenate((x, df.reshape(1, -1)), axis=0)

        y = np.empty((0, label.shape[1]))
        for id, la in zip(person_id, label):
            if id in train_ids:
                y = np.concatenate((y, la.reshape(1, -1)), axis=0)
        train_data.append(x)
        train_label.append(y)

        x = np.empty((0, data.shape[1]))
        for id, df in zip(person_id, data):
            if id in test_ids:
                x = np.concatenate((x, df.reshape(1, -1)), axis=0)

        p = np.empty((0, 1))
        y = np.empty((0, label.shape[1]))
        f = np.empty((0, 1))
        for id, la, fin in zip(person_id, label, file_name):
            if id in test_ids:
                y = np.concatenate((y, la.reshape(1, -1)), axis=0)
                p = np.concatenate((p, np.full((1, 1), id)), axis=0)
                f = np.concatenate((f, np.full((1, 1), fin)), axis=0)
        test_data.append(x)
        test_label.append(y)
        person_id_list.append(p)
        file_list.append(f)

    return train_data, test_data, train_label, test_label, person_id_list, file_list

train_x, test_x, train_y, test_y, personidfold, filenamelist = split_data_kfold(person_id, file_name_list, combined_data, one_hot_labels, num_folds)
total_result = []

for fold in range(num_folds):
    # create training and testing dataset
    print(f'------------------------{fold} fold---------------------------')
    test_data = test_x[fold]
    test_labels = test_y[fold]

    train_data = train_x[fold]
    train_labels = train_y[fold]

    test_id = personidfold[fold]
    test_file_name = filenamelist[fold]


    import torch.nn as nn

    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)  # One hidden layer
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)  # Output layer

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            return x

    input_size = combined_data.shape[1]
    hidden_size = 32  # Set the size of the hidden layer
    num_classes = 3  # Set the number of output classes

    model = SimpleMLP(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    num_epochs = 100
    batch_size = 16

    for epoch in range(num_epochs):
        permutation = np.random.permutation(train_data.shape[0])
        shuffled_data = train_data[permutation]
        shuffled_labels = train_labels[permutation]

        for i in range(0, train_data.shape[0], batch_size):
            inputs = torch.from_numpy(shuffled_data[i:i+batch_size]).float()
            target = torch.argmax(torch.from_numpy(shuffled_labels[i:i+batch_size]), dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(test_data).float()
        target = torch.argmax(torch.from_numpy(test_labels), dim=1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == target).sum().item() / test_data.shape[0]
        accuracies.append(accuracy)
        print(f'Fold [{fold}/{num_folds}] Accuracy (video): {accuracy}')

        test_df = pd.DataFrame(np.array([test_id.flatten().tolist(),
                                         test_file_name.flatten().tolist(),
                                         predicted.numpy().flatten().tolist(),
                                         target.numpy().flatten().tolist()]).T,
                               columns=['person_id', 'file_name', 'pred_label', 'true_label'])

        person_result = test_df.groupby('person_id').agg({'pred_label': pd.Series.mode,
                                                          'true_label': pd.Series.mode})
        person_result['pred_label'] = person_result['pred_label'].apply(lambda x: x if type(x) == str else x[-1])
        person_result['true_label'] = person_result['true_label'].apply(lambda x: x if type(x) == str else x[-1])
        print(f"Fold [{fold}/{num_folds}] Accuracy (person): {accuracy_score(person_result['true_label'], person_result['pred_label'])}")

        if len(total_result) == 0:
            total_result = person_result
        else:
            total_result = pd.concat([total_result, person_result])

        print(f'{fold} fold report')

        print("Target Labels:", target.numpy().flatten().tolist())
        print("Predicted Labels:", predicted.numpy().flatten().tolist())

        print(classification_report(target.numpy().flatten().tolist(),
                                    predicted.numpy().flatten().tolist(),
                                    target_names=["Normal", "PD-Mild", "PD-Moderate"],digits=4))
        print(f"AUC for each label and overall{multiclassauc(target.numpy().flatten().tolist(),predicted.numpy().flatten().tolist(), n_classes=3)}")


mean_accuracy = np.mean(accuracies)
print(f'Mean Accuracy: {mean_accuracy}')
print('Classification Reports For 81 Subjects')
print(classification_report(total_result['true_label'], total_result['pred_label'], target_names=["Normal", "PD-Mild", "PD-Moderate"], digits=4), )
print('Raw Data to Contain the Performance Information')
# print(total_result)

print(total_result.shape)

from sklearn.preprocessing import LabelEncoder

# Initialize a LabelEncoder
label_encoder = LabelEncoder()

# Fit the encoder on your labels
label_encoder.fit(total_result['true_label'])

# Transform the true and predicted labels
total_result['true_label'] = total_result['true_label'].apply(lambda x: x if x in label_encoder.classes_ else 'Unknown')
total_result['pred_label'] = total_result['pred_label'].apply(lambda x: x if x in label_encoder.classes_ else 'Unknown')

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Extract 'pred_label' and 'true_label' columns for a multi-class classification
y_pred = total_result['pred_label']
y_true = total_result['true_label']

# Calculate the confusion matrix
confusion = confusion_matrix(y_true, y_pred)

# Create a more visually appealing heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", square=True, cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

n_classes = len(total_result['true_label'].unique())

plt.figure(figsize=(10, 8))

for i in range(n_classes):
    fpr, tpr, thresholds = roc_curve((y_true == i).astype(int), (y_pred == i).astype(int))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC class {i} = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Calculate and print the classification report
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, digits=4)
print("Classification Report:\n", report)

plt.show()