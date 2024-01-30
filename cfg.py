class ConfigData():
    # path
    img_folder_root = r'D:\BackendAOI\Data\PassNg_Images\0_A73A\4HW3-2802-TR1C_(0)\FM' #50
    output_root = r'D:\BackendAOI\Python\cnn\save_model\1.A73A_CNN_model\4HW3-2802-TR1C_Fm_V3'
    output_name = r'2class'
    pretrained_weights = r''
    # param for data
    valid_keep_ratio = 0.2
    img_newsize = 400
    device_memory_ratio = 0.99
    workers = 1
    epochs = 300
    early_stop = 80
    batch_size = 32
    num_classes = 2
    over_sampling_thresh = 1000
    over_sampling_scale = 2
    

class HypScratch():
    lr = 0.001
    lrf = 0.2
    momentum = 0.925
    weight_decay = 0.0004
    warmup_epochs = 3
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1


