
import numpy as np


class FindChangePts():
    def __init__(self,x_array, survtime_array, event_flag_array):
        self.changePt_threshold = 0.05
        # self.changePt_threshold: the field  prevents the algorithm detecting a change point at both extremes
        # if you don't want to use this function, block the code lines 101~102
        self.num_max_change_pts = np.inf
        # self.num_max_change_pts = np.inf (float):
        # if there is a desired upper bound of change_pts, a researcher can customize this parameter.
        self.max_LogLikelihood = cal_maxLogLikelihood_exp_twoPara
        self.nParams = 2
        """
        if a researcher want to change/customize data model, 
        "implement her own function" which yields a maximum log likelihood and assign it to
        "self.max_LogLikelihood"
        and also assign the number of parameters of the data model to
        "self.nParams"
        """
        self.changePoints_array = None
        """
        the field array containing detected change points 
        """
        self.find_change_points(np.copy(x_array), [np.copy(survtime_array), np.copy(event_flag_array)])

    def find_change_points(self, x_array, y_array_list):
        """
        :param x_array (numpy array, float): a predictor variable
        :param y_array_list = [survtime_array, event_flag_array]
        """
        def isnan(input_data):
            try:
                return np.isnan(input_data)
            except:
                return False
        masking_array = ~(isnan(x_array) | isnan(y_array_list[0]))
        x_array = x_array[masking_array]
        y_array_list = [elt[masking_array] for elt in y_array_list]

        sorted_index = np.argsort(x_array)
        x_array_sorted = x_array[sorted_index]
        y_list_sorted = [elt[sorted_index] for elt in y_array_list]
        changePointList = []
        segIntv_list = [(0, len(x_array_sorted) - 1)]
        segIntv2indexScore_dict = {(0, len(x_array_sorted) - 1): []}
        count = 0
        while count < self.num_max_change_pts:
            max_score = float('-inf')
            max_index = -1  # None
            max_breakPt_index = -1  # None
            for index, partitionData in enumerate(segIntv_list):
                if not segIntv2indexScore_dict[partitionData]:
                    y_list_sub = [elt[partitionData[0]:partitionData[1]] for elt in y_list_sorted]
                    (breakPt_index, breakPtScore) = self.find_a_change_point(y_list_sub)
                    segIntv2indexScore_dict[partitionData].append((breakPt_index, breakPtScore))
                breakPt_index, breakPtScore = segIntv2indexScore_dict[partitionData][0]
                if breakPtScore > max_score:
                    max_score = breakPtScore
                    max_index = index
                    max_breakPt_index = partitionData[0] + breakPt_index
            if max_breakPt_index >= 0:
                if max_breakPt_index in changePointList:
                    break
                changePointList.append(max_breakPt_index)
            else:
                break
            toBeRemovedPartition = segIntv_list[max_index]
            segIntv_list.remove(toBeRemovedPartition)
            del segIntv2indexScore_dict[toBeRemovedPartition]

            firstPartitionTuple = (toBeRemovedPartition[0], max_breakPt_index)
            secondPartitionTuple = (max_breakPt_index, toBeRemovedPartition[1])
            segIntv_list.append(firstPartitionTuple)
            segIntv_list.append(secondPartitionTuple)
            segIntv2indexScore_dict[firstPartitionTuple] = []
            segIntv2indexScore_dict[secondPartitionTuple] = []
            count += 1
        changePointList.sort()
        self.changePoints_array = np.unique(x_array_sorted[changePointList])

    def find_a_change_point(self, y_list_sub):
        censored_data_list = y_list_sub
        nSample = len(censored_data_list[0])
        LL_0 = self.max_LogLikelihood(censored_data_list)
        SIC_n = -2 * LL_0 + self.nParams * np.log(nSample)
        SIC_list = []
        for k in range(1, nSample - 1):
            head_data = [elt[:k] for elt in censored_data_list]
            tail_data = [elt[k:] for elt in censored_data_list]
            LL_H = self.max_LogLikelihood(head_data)
            LL_T = self.max_LogLikelihood(tail_data)
            SIC_K = -2 * (LL_H + LL_T) + 2 * self.nParams * np.log(nSample)
            SIC_list.append(SIC_K)
        SIC_list_combined = SIC_list + [SIC_n]
        count = 0
        sorted_index = np.argsort(SIC_list_combined)
        min_index = sorted_index[count]

        if (min_index <= nSample * self.changePt_threshold) or (min_index >= nSample * (1-self.changePt_threshold)):
            return -1, float('-inf')
        # If null Hypothesis is not rejected
        if min_index == (nSample-2):
            return -1, float('-inf')
        # else
        min_SIC = SIC_list_combined[min_index]
        index = min_index
        score = SIC_n
        # instead, you can also use this code: score = SIC_n / min_SIC
        return index, score


def cal_maxLogLikelihood_exp_twoPara(censored_data):
    completeData = censored_data[0][censored_data[1] == 1]
    censoredData = censored_data[0][censored_data[1] == 0]

    if len(completeData) == 0:
        return np.nan
    t_1 = np.min(completeData)
    sum_completedTime = np.sum(completeData)
    sum_censoredTime = np.sum(censoredData)
    r = len(completeData)
    nSample = r + len(censoredData)
    lam_est = (r - 1) / (sum_completedTime + sum_censoredTime - nSample * t_1)
    G_est = max([t_1 - 1 / (nSample * lam_est), 0])
    LL = r * np.log(lam_est) - lam_est * (sum_completedTime + sum_censoredTime - nSample * G_est)
    return LL
