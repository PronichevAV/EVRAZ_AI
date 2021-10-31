import numpy as np
import datetime

def get_timestamp():
    # Возвращает временную метку дата + время
    timestamp = datetime.datetime.now().isoformat(' ', 'minutes')  # without microseconds
    timestamp = timestamp.replace(' ', '-').replace(':', '-')
    return timestamp

def metric_C(true, pred):

    delta_c = np.abs(true - pred)
    hit_rate_c = np.int64(delta_c < 0.02)

    N = np.size(true)

    return np.sum(hit_rate_c) / N

def metric_TST(true, pred):
    delta_t = np.abs(true - pred)
    hit_rate_t = np.int64(delta_t < 20)

    N = np.size(true)

    return np.sum(hit_rate_t) / N

    