import numpy as np
import torch

from anfis_pytorch import membership


class RTDataPreprocessing:
    def __init__(self, sampling_time, normalization_parameters):
        self.sampling_time = sampling_time
        self.normalization_parameters = normalization_parameters
        self.steering_w_angle_prev = 0.0
        self.lat_accel_prev = 0.0
        self.yaw_rate_prev = 0.0

    def __call__(self, steering_w_angle, lat_accel, yaw_rate, velocity=None):
        steering_w_rate = self.calculate_derivative(steering_w_angle, self.steering_w_angle_prev)
        lat_jerk = self.calculate_derivative(lat_accel, self.lat_accel_prev)
        yaw_accel = self.calculate_derivative(yaw_rate, self.yaw_rate_prev)
        x_signals = {
            'SteeringWheelAngle': steering_w_angle,
            'steering_wheel_rate': steering_w_rate,
            'YawRate': yaw_rate,
            'yaw_acceleration': yaw_accel,
            'lateral_jerk': lat_jerk
        }
        if velocity is not None:
            x_signals["VelocityX"] = velocity
        x_normalized = {}
        for key in x_signals.keys():
            x_normalized[key] = self.normalize(key, x_signals[key], self.normalization_parameters)
        self.steering_w_angle_prev = steering_w_angle
        self.lat_accel_prev = lat_accel
        self.yaw_rate_prev = yaw_rate
        return x_normalized

    def calculate_derivative(self, value, value_prev):
        derivative = (value - value_prev) / self.sampling_time
        return derivative

    def normalize(self, signal_name, value, normalization_parameters):
        method = normalization_parameters[signal_name]['method']
        p1 = normalization_parameters[signal_name]['p1']
        p2 = normalization_parameters[signal_name]['p2']
        if method == 'minmax':
            output = self.normalize_minmax(value, p1, p2)
        elif method == 'scaling':
            output = self.normalize_scaling(value, p1, p2)
        elif method == 'zscore':
            output = self.normalize_zscore(value, p1, p2)
        elif method == 'minmax_positive':
            output = self.normalize_minmax_positive(value, p1, p2)
        else:
            raise f"Unknown normalization method '{method}'"
        return output

    def normalize_minmax(self, value, min, max):
        return (value - min) / (max - min)

    def normalize_scaling(self, value, min, max):
        if min < 0 and max > 0:
            output = value / np.abs(max) if value > 0 else value / np.abs(min)
        else:
            output = self.normalize_minmax(value)
        return output

    def normalize_zscore(self, value, mu, std):
        return (value - mu) / std

    def normalize_minmax_positive(self, value, min, max):
        return self.normalize_minmax(value, min, max)

class RT_OS_US_Inference:
    def __init__(self, data_preprocessor, torch_model):
        self.data_preprocessor = data_preprocessor
        self.torch_model = torch_model
        self.sampling_time = self.data_preprocessor.sampling_time

    def __call__(self, steering_w_angle, lat_accel, yaw_rate, velocity=None):
        x_normalized_dict = self.data_preprocessor(steering_w_angle, lat_accel, yaw_rate, velocity)
        x = dict_to_tensor(x_normalized_dict)
        return infer_indicator(self.torch_model, x)


def load_anfis_model(weights_path, varnums, mfs_nums):
    dummy_x = torch.Tensor([0] * varnums).unsqueeze(0)
    model = membership.make_anfis(dummy_x, num_mfs=mfs_nums, num_out=1, hybrid=False)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def infer_indicator(model, x):
    y_pred = model(x)
    y_pred = y_pred.squeeze()
    return y_pred.item()


def open_json(path):
    import json
    with open(path) as f:
        d = json.load(f)
    return d


def dict_to_tensor(x_dict):
    values = []
    for value in x_dict.values():
        values.append(value)
    x = torch.FloatTensor(values).unsqueeze(0)
    return x
