from FindChangePts import *

_NUM_SAMPLE = 1000
_TRUE_CHANGE_POINT = 507


def main():
    x, surv_time = gen_sample_data(_TRUE_CHANGE_POINT, _NUM_SAMPLE)
    nSample = surv_time.shape[0]
    event_flag_array = np.ones(nSample)
    censored_idx_array = np.random.choice(nSample, int(nSample*0.2), replace=False)
    event_flag_array[censored_idx_array] = 0
    ChangePts = FindChangePts(x, surv_time, event_flag_array)
    print("True change point:", _TRUE_CHANGE_POINT)
    print("Estimated change point:", ChangePts.changePoints_array + 1)



def gen_sample_data(changePtIndex,nSample):
    """
    :param changePtIndex: the index of change point index
    :param nSample: number of sample, nSample > changePtIndex
    :return x: sample predictor array
    :return y: sampled response array
    """
    def gen_two_param_exp_data(constant_rate, location, n_sample):
        x = np.random.uniform(size=n_sample)
        return np.log(1 - x) / (-constant_rate) + location

    if changePtIndex >= nSample:
        raise ValueError("changePtIndex must be smaller than nSample")
    head_data_array = gen_two_param_exp_data(constant_rate=1, location=0, n_sample=changePtIndex)
    tail_data_array = gen_two_param_exp_data(constant_rate=5,location=3, n_sample=(nSample-changePtIndex))
    y = np.concatenate((head_data_array,tail_data_array))
    x = np.arange(y.shape[0])
    return x, y


if __name__ == "__main__":
   main()
