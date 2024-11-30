import torch
import csv
from model import MLP

model_mlp = MLP(17)
model_mlp.load_state_dict(torch.load("property2/mlp.pt"))
# param = torch.nn.parameter(model_mlp._MLP__fc1)

first = model_mlp._MLP__fc1.weight.data
first_b = model_mlp._MLP__fc1.bias.data

second = model_mlp._MLP__fc2.weight.data
second_b = model_mlp._MLP__fc2.bias.data

third = model_mlp._MLP__fc3.weight.data
third_b = model_mlp._MLP__fc3.bias.data

fourth = model_mlp._MLP__fc4.weight.data
fourth_b = model_mlp._MLP__fc4.bias.data

weight_list = [[first, first_b], [second, second_b], [third, third_b], [fourth, fourth_b]]

input = torch.ones(1, 17)
output = model_mlp._MLP__fc1(input)
# print(output[0][0])
# print(torch.sum(first[0, :]) + first_b[0])

save_weight = []
layer_size = [17, 128, 64, 32, 1]

model_weight = []
bias_weight = []
for ln in range(0, len(layer_size) - 1): # 4
    layer_weight = []
    bias = []

    for out_layer in range(0, layer_size[ln + 1]):
        node_layer = []
        for in_layer in range(0, layer_size[ln]):
            wgt = weight_list[ln][0][out_layer, in_layer].item()
            node_layer.append(wgt)

        layer_weight.append(node_layer)
        bias.append(weight_list[ln][1][out_layer].item())
    
    bias_weight.append(bias)
    model_weight.append(layer_weight)
    
print(len(model_weight[0]), len(model_weight[0][0]), len(model_weight[1]), len(model_weight[1][0]), 
      len(model_weight[2]), len(model_weight[2][0]), len(model_weight[3]), len(model_weight[3][0]))
print(len(bias_weight[0]), len(model_weight[1]), 
      len(bias_weight[2]), len(model_weight[3]))

# weight_csv = open("mlp_weight.csv", "w", newline='')
# weight_writer = csv.writer(weight_csv)
# # with open('mlp_weight.csv', 'w', newline='', encoding='utf-8') as weight_csv:
# #     weight_writer = csv.writer(weight_csv)

# weight_writer.writerow("layer1")
# weight_writer.writerows(model_weight[0])
# weight_writer.writerow("\nlayer2")
# weight_writer.writerows(model_weight[1])
# weight_writer.writerow("\nlayer3")
# weight_writer.writerows(model_weight[2])
# weight_writer.writerow("\nlayer4")
# weight_writer.writerows(model_weight[3])
# weight_csv.close()
import struct

f = open("mlp_weight.bin", "wb")
f.write(b"ETRI CGM MLP WGT V1.0")
layers = struct.pack('<hhhhhh', 5, 17, 128, 64, 32, 1)
f.write(layers)
for output_layer in range(0, 128):
    for input_layer in range(0, 17):
        weight_part = struct.pack('<f', model_weight[0][output_layer][input_layer])
        f.write(weight_part)

for output_layer in range(0, 128):
    bias_part = struct.pack('<f', bias_weight[0][output_layer])
    f.write(bias_part)

for output_layer in range(0, 64):
    for input_layer in range(0, 128):
        weight_part = struct.pack('<f', model_weight[1][output_layer][input_layer])
        f.write(weight_part)

for output_layer in range(0, 64):
    bias_part = struct.pack('<f', bias_weight[1][output_layer])
    f.write(bias_part)

for output_layer in range(0, 32):
    for input_layer in range(0, 64):
        weight_part = struct.pack('<f', model_weight[2][output_layer][input_layer])
        f.write(weight_part)

for output_layer in range(0, 32):
    bias_part = struct.pack('<f', bias_weight[2][output_layer])
    f.write(bias_part)

for output_layer in range(0, 1):
    for input_layer in range(0, 32):
        weight_part = struct.pack('<f', model_weight[3][output_layer][input_layer])
        f.write(weight_part)

for output_layer in range(0, 1):
    bias_part = struct.pack('<f', bias_weight[3][output_layer])
    f.write(bias_part)

mul_part = struct.pack('<f', 1000.0)
f.write(bias_part)

f.close()