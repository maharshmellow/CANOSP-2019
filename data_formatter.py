import csv
import numpy as np 

def load_data(dataset):
    data = {}

    with open(dataset) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers

        for row in reader:
            user_id = int(row[-1])
            
            features = [float(feature) for feature in row[1:5]]
            label = int(row[5])

            if user_id not in data:
                data[user_id] = []
            
            data[user_id].append([features, label])

    num_clients = len(data)

    features = [[] for i in range(num_clients)] # (num_clients, num_samples, num_features)  [ [[1, 1], [2, 2]] , [[3, 3], [4, 4]] ]
    labels = [[] for i in range(num_clients)]   # (num_clients, num_samples)                  [ [1, 2], [3, 4] ]

    for user_id, samples in data.items():
        for sample in samples:
            features[user_id].append(sample[0])
            labels[user_id].append(sample[1])

    features = np.array(features)
    labels = np.array(labels)

    return features, labels

features, labels = load_data("datasets/blob_S20000_L3_F4_U100.csv")
print(features.shape)
print(labels.shape)
