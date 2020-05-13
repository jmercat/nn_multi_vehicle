import torch
from loaders.loadMultiObjectFusion import MultiObjectFusionDataset
from loaders.loadMultiObjectNGSIM import MultiObjectNGSIMDataset
from loaders.loadMultiObjectArgoverse import MultiObjectArgoverseDataset
from models.Conv import conv_model
from models.FC import FC_model
from models.RNN import LSTM_model, LSTM_model2
import os, sys
import numpy as np
import yaml


class Settings:
    class __Settings:
        def __init__(self):
            self.settings_dict = yaml.safe_load(open('utils/settings.yaml'))
            self.refresh()

        def refresh(self):
            if self.settings_dict['device'] == '':
                self.settings_dict['device'] = 'cpu'
                if torch.cuda.is_available():
                    self.settings_dict['device'] = 'cuda'
                    print('Using device ' + torch.cuda.get_device_name())
            self.settings_dict['use_yaw'] = self.settings_dict['model_type'][:7] == 'Bicycle'
            self.settings_dict['name'] = (self.settings_dict['model_type'] + '_' +
                                          self.settings_dict['dataset'] + '_' +
                                          str(self.settings_dict['training_id']))
            self.settings_dict['log_path'] = ('./logs/' )
            self.settings_dict['models_path'] = ('./trained_models/')
            if self.settings_dict['dataset'] == 'NGSIM':
                self.settings_dict['dt'] = 0.1*self.settings_dict['down_sampling']
                self.settings_dict['unit_conversion'] = 0.3048
                self.settings_dict['time_hist'] = min(3, self.settings_dict['time_hist'])
                self.settings_dict['time_pred'] = min(5, self.settings_dict['time_pred'])
                self.settings_dict['field_height'] = 30
                self.settings_dict['field_width'] = 120
                self.settings_dict['n_max_veh'] = ((self.settings_dict['n_max_veh'] + 2) // 3) * 3
            elif self.settings_dict['dataset'] == 'Argoverse':
                self.settings_dict['dt'] = 0.1*self.settings_dict['down_sampling']
                self.settings_dict['unit_conversion'] = 1
                self.settings_dict['time_hist'] = 2
                self.settings_dict['time_pred'] = min(3, self.settings_dict['time_pred'])
                self.settings_dict['field_height'] = 120
                self.settings_dict['field_width'] = 120
            elif self.settings_dict['dataset'] == 'Fusion':
                self.settings_dict['dt'] = 0.04*self.settings_dict['down_sampling']
                self.settings_dict['unit_conversion'] = 1
                self.settings_dict['time_hist'] = 2
                self.settings_dict['time_pred'] = min(3, self.settings_dict['time_pred'])
                self.settings_dict['field_height'] = 120
                self.settings_dict['field_width'] = 120
            else:
                raise ValueError('The dataset "' + self.settings_dict['dataset'] + '" is unknown. Please correct the'
                                 'dataset name in "settings.yaml" or modify the Settings class in "utils.py" to handle it.')

        def __str__(self):
            return repr(self) + self.settings_dict

    instance = None
    def __init__(self):
        if not Settings.instance:
            Settings.instance = Settings.__Settings()
        else:
            pass
    def __getattr__(self, name):
        return self.instance.settings_dict[name]

    def __setattr__(self, name, value):
        self.instance.settings_dict[name] = value
        self.instance.refresh()

    def get_dict(self):
        return self.instance.settings_dict.copy()



def sort_predictions(pred_fut):
    pred_fut[:, :, :, :, 5] = np.mean(pred_fut[:, :, :, :, 5], axis=0, keepdims=True)
    flat_pred_test = pred_fut.reshape([-1, 6, 6])
    flat_argsort_p = np.argsort(flat_pred_test[:, :, 5], axis=1)[:, ::-1]
    flat_pred_test_sorted_p = flat_pred_test.copy()
    for i in range(6):
        flat_pred_test_sorted_p[:, i, :] = flat_pred_test[np.arange(flat_pred_test.shape[0]), flat_argsort_p[:, i]]
    return flat_pred_test_sorted_p.reshape(
        [pred_fut.shape[0], pred_fut.shape[1], pred_fut.shape[2], pred_fut.shape[3], pred_fut.shape[4]])


def get_multi_object_dataset():
    args = Settings()
    if args.dataset == 'NGSIM':
        trSet = MultiObjectNGSIMDataset(args.NGSIM_data_directory + 'TrainSet_traj_v2.mat',
                                        args.NGSIM_data_directory + 'TrainSet_tracks_v2.mat', args=args)
        valSet = MultiObjectNGSIMDataset(args.NGSIM_data_directory + 'ValSet_traj_v2.mat',
                                         args.NGSIM_data_directory + 'ValSet_tracks_v2.mat', args=args)
    elif args.dataset == 'Argoverse':
        trSet = MultiObjectArgoverseDataset(args.argoverse_data_directory + 'train/data', args=args)
        valSet = MultiObjectArgoverseDataset(args.argoverse_data_directory + 'val/data', args=args)
    elif args.dataset == 'Fusion':
        trSet = MultiObjectFusionDataset(args.fusion_data_directory + 'train_sequenced_data.tar', args=args)
        valSet = MultiObjectFusionDataset(args.fusion_data_directory + 'val_sequenced_data.tar', args=args)

    return trSet, valSet


def get_multi_object_test_set():
    args = Settings()
    if args.dataset == 'NGSIM':
        testSet = MultiObjectNGSIMDataset(args.NGSIM_test_data_directory + 'TestSet_traj_v2.mat',
                                          args.NGSIM_test_data_directory + 'TestSet_tracks_v2.mat', args)
    elif args.dataset == 'Fusion':
        testSet = MultiObjectFusionDataset(args.fusion_data_directory + 'test_sequenced_data.tar', args)
    elif args.dataset == 'Argoverse':
        testSet = MultiObjectArgoverseDataset(args.argoverse_data_directory + '/val/dataset2/', False, False, True)
    else:
        raise RuntimeError('Multi object loader does not support other datasets than NGSIM and Fusion.')
    return testSet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_multi_object_net():
    args = Settings()

    if args.model_type == 'FC':
        net = FC_model(args)
    elif args.model_type == 'conv':
        net = conv_model(args)
    elif args.model_type == 'LSTM':
        net = LSTM_model(args)
    elif args.model_type == 'LSTM2':
        net = LSTM_model2(args)
    else:
        print('Model type ' + args.model_type + ' is not known.')

    net = net.to(args.device)
    print("Net number of parameters: %d" % count_parameters(net))

    if args.load_name != '':
        try:
            net.load_state_dict(torch.load('./trained_models/' + args.model_type + '/' + args.load_name + '.tar', map_location=args.device))
        except RuntimeError as err:
            print(err)
            print('Loading what can be loaded with option strict=False.')
            net.load_state_dict(
                torch.load('./trained_models/' + args.model_type + '/' + args.load_name + '.tar',
                           map_location=args.device), strict=False)
    return net

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass
