from django.http import JsonResponse

from modelVisualization.services.SeedVisualization.data_loader import *
from modelVisualization.services.SeedVisualization.eeg_service import *
from modelVisualization.services.SeedVisualization.psd_service import *


current_trials = []


def datasets(request):

    return JsonResponse(list_datasets(),safe=False)


def mat_files(request,dataset):

    return JsonResponse(list_mat_files(dataset),safe=False)


def load_mat(request,dataset,filename):

    global current_trials

    mat = load_mat(dataset,filename)

    current_trials = extract_trials(mat)

    return JsonResponse({"trials":len(current_trials)})


def channel_series(request,trial,channel):

    t = current_trials[trial]

    s = get_channel_series(t,channel)

    return JsonResponse(s.tolist(),safe=False)


def psd(request,trial,channel):

    t = current_trials[trial]

    s = get_channel_series(t,channel)

    f,p = compute_psd(s)

    return JsonResponse({"freq":f,"power":p})