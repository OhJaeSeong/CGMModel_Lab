import os
import json
import yaml

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from base_densenet import DenseNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(f'{BASE_DIR}/environments_ofDenseNet.yaml', 'r') as f:  # load yaml file
    environment = yaml.load(f, Loader=yaml.FullLoader)

property = environment['densenet_model']
user_setting = environment['user_setting']
address_book = environment['address_book']

reduceLR_term = user_setting['reduceLR_term']
majino_accuracy = user_setting['minimum_demand_accuracy']
save_step = user_setting['step']
minEpoch_toSave = user_setting['minEpoch_toSave']

class DenseNet121:
    def __init__(self, category_count):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(property['config_file'], "r") as i:
            densenet_cfg = json.load(i)

        self.num_class = category_count
        growth_rate = densenet_cfg["growth_rate"]
        block_config = tuple(densenet_cfg["block_config"])
        num_init_features = densenet_cfg["num_init_features"]

        # Prepare Model
        print("Loading DenseNet121 network.....")
        model = DenseNet(growth_rate, block_config, num_init_features)
        num_features = model.classifier.in_features

        model.classifier = None

        '''가중치 FC부분 제거'''
        weight_file = torch.load(property['model_weight_file'], map_location=self.device)
        if 'classifier.weight' in weight_file:
            del weight_file['classifier.weight']

        if 'classifier.bias' in weight_file:
            del weight_file['classifier.bias']

        model.load_state_dict(weight_file)
        model.eval()

        print("Network successfully loaded")

        self.preprocess = transforms.Compose([
            transforms.Resize(densenet_cfg["resize_value"]),
            transforms.CenterCrop(densenet_cfg["center_crop_value"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=densenet_cfg["normalize_mean"],
                                 std=densenet_cfg["normalize_std"]),
        ])

        classifier = nn.Linear(num_features, self.num_class)
        if property['classifier_weight_file'] != '':
            if os.path.isfile(property['classifier_weight_file']):
                classifier.load_state_dict(torch.load(property['classifier_weight_file'],
                                                      map_location=self.device))
                print(f"saved {property['classifier_weight_file']} equipped")

        model.classifier = classifier

        model.to(self.device)
        self.model = model

        with open(property['labels'], "r") as i:
            self.label_index = json.load(i)

    def train_classifier(self, learning_rate=0.00005, epoch=10, batchs=8):
        train_data = torchvision.datasets.ImageFolder(root=address_book['train_address'],
                                                      transform=self.preprocess)

        data_loader = DataLoader(train_data, batch_size=batchs, shuffle=True, drop_last=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        # object function, optimizer setting

        total_len = len(train_data)
        loss_sum = 0
        count = 0
        percentage_ten = 0.1
        # variable for interim checks

        models_score = majino_accuracy

        print("classifier training start")

        for i in range(epoch):
            for batch_idx, samples in enumerate(data_loader):
                x, y = samples

                x = x.to(self.device)
                y = y.to(self.device)

                prediction = self.model(x)
                cost = criterion(prediction, y)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                # each dataset(bound in batch size), doing backward to train

                loss_sum += cost
                count += batchs
                if count > int(total_len * percentage_ten) and count != 0 and percentage_ten != 1.0:
                    completed = count / total_len
                    print(round(completed * 100, 2), '% complete in epoch ', i + 1)
                    percentage_ten += 0.1
                # print output current status at 10% completion per each epoch
                if (count % (batchs * save_step)) == 0 and i >= minEpoch_toSave:
                    print("You just activate my val Card")
                    models_score = self.validate_and_save(models_score, i + 1, int(count / batchs))
                    # in more than three epoch statem every 100 step, save weight file temporary

            print(f'avg loss : {loss_sum / count}')
            print(f'{i + 1} epoch cleared.')
            # pring average cost(loss) and current epoch

            loss_sum = 0
            count = 0
            percentage_ten = 0.1

            if (i + 1) % reduceLR_term == 0 and i != 0:
                learning_rate = self.reduce_LR(learning_rate)
                optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
                # for prevent overfitting, adaptive learning rate used

        self.validate_and_save(models_score, epoch, 0)