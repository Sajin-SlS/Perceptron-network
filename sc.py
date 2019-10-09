import csv

def data_to_float(row):
    return list(map(lambda x: float(x),row))

def dataset_read(filepath):
	converted_set = list()
	reader = csv.reader(open(filepath,'r') )
	for row in reader:
		converted_set.append(data_to_float(row))
	return converted_set

def pred_val(weights,inputs,threshold):
    fire = weights[0]
    for i in range(len(weights)-1):
        fire += weights[i+1] * inputs[i]
    return 1.0 if fire >= threshold else 0.0

dataset = dataset_read('Cryotherapy.csv')
weights = [1.0 for x in range(len(dataset[0]))]
learning_rate = float(input("Provide learning rate for the network: "))
threshold = float(input("Provide threshold value for the network: "))
epoch = int(input("Provide the number of epochs should the training do: "))

for x in range(epoch):
    for data in dataset:
        calculated_value = pred_val(weights,data,threshold)
        if calculated_value != data[-1]:
            error = data[-1] - calculated_value
            weights[0] += learning_rate * error
            for i in range(len(weights)-1):
                weights[i+1] += data[i] * learning_rate * error

print("Weights calculated: ",weights)