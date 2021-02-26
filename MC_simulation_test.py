from FindChangePts import *
"""
 Global Variables user can customize
 _NUM_SAMPLE : total number of sample
 _TRUE_CHANGE_POINT_LIST: change point list
 _CENSORING_RATE: censoring rate among total sample
 
 * remark
 the largest value of _TRUE_CHANGE_POINT_LIST (max(_TRUE_CHANGE_POINT_LIST)) must be
 smaller than _NUM_SAMPLE
 
"""
_NUM_SAMPLE = 7000
_TRUE_CHANGE_POINT_LIST = [507, 1200, 1995, 2500, 3000]
_CENSORING_RATE = 0.25


def main():
    x, surv_time_array = gen_sample_data(_TRUE_CHANGE_POINT_LIST, _NUM_SAMPLE)
    nSample = surv_time_array.shape[0]
    event_flag_array = np.ones(nSample)
    censored_idx_array = np.random.choice(nSample, int(nSample*_CENSORING_RATE), replace=False)
    # gen censoring
    for ix in censored_idx_array:
        event_flag_array[ix] = 0
        surv_time = surv_time_array[ix]
        surv_time_array[ix] = np.random.uniform(0,surv_time)

    ChangePts = FindChangePts(x, surv_time_array, event_flag_array)
    print("True change point:", _TRUE_CHANGE_POINT_LIST)
    print("Estimated change point:", ChangePts.changePoints_array + 1)



def gen_sample_data(changePtIndex_list,nSample):
    """
    :param changePtIndex_list: list of change point indexes
    :param nSample: number of sample, nSample > changePtIndex
    :return x: sample predictor array
    :return y: sampled response array
    """
    def gen_two_param_exp_data(constant_rate, location, n_sample):
        x = np.random.uniform(size=n_sample)
        return np.log(1 - x) / (-constant_rate) + location

    changePtIndex_list = sorted(changePtIndex_list)
    if changePtIndex_list[-1] >= nSample:
        raise ValueError("changePtIndex must be smaller than nSample")
    data_list = list()
    change_pt_ix_prev = 0
    for i, change_pt_ix in enumerate(changePtIndex_list):
        head_data_array = gen_two_param_exp_data(constant_rate=(i+1), location=i, n_sample=(change_pt_ix-change_pt_ix_prev))
        data_list.append(head_data_array)
        change_pt_ix_prev = change_pt_ix
    data_list.append(gen_two_param_exp_data(constant_rate=i+2, location=i+1, n_sample=(nSample-change_pt_ix)))
    y = np.concatenate(data_list)
    x = np.arange(y.shape[0])
    return x, y


if __name__ == "__main__":
   main()