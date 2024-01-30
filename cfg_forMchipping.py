
class ConfigData():
    # path
    img_folder_root = r'D:\BackendAOI\A_Training\4KL7-2801-TR1C_(V1)\Bin2-chipping\1_trainingdata' #50
    output_root = r'D:\BackendAOI\A_Training\4KL7-2801-TR1C_(V1)\Bin2-chipping\4_model'
    output_name = r'10class'
    pretrained_weights = r''

    # param for data
    valid_keep_ratio = 0.2
    img_newsize = 400
    device_memory_ratio = 0.99
    workers = 1
    epochs = 300
    early_stop = 60
    batch_size = 32
    num_classes = 10
    over_sampling_thresh = 1000
    over_sampling_scale = 2
    

class HypScratch():
    lr = 0.0005 #0.001
    lrf = 0.1 #0.2
    momentum = 0.925
    weight_decay = 0.0004
    warmup_epochs = 3
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1


