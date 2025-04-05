import os

ROOT_DATASET = ''

# ./video_frames: Your video dataset

def return_ucf101():
    filename_categories = './data/ucf101/classInd.txt'
    root_data = './video_frames/UCF101/jpg'
    filename_imglist_train = './data/ucf101/ucf101_rgb_train_list.txt'
    filename_imglist_val = './data/ucf101/ucf101_rgb_val_list.txt'
    filename_imglist_balance = './data/ucf101/balance.txt'
    prefix = 'img_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, filename_imglist_balance, root_data, prefix


def return_hmdb51():
    filename_categories = './data/hmdb51/ClassInd.txt'
    root_data = './video_frames/HMDB51/jpg'
    filename_imglist_train = './data/hmdb51/hmdb51_train_split_1_rawframes.txt'
    filename_imglist_val = './data/hmdb51/hmdb51_val_split_1_rawframes.txt'
    filename_imglist_balance = './data/hmdb51/balance.txt'
    prefix = 'img_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, filename_imglist_balance, root_data, prefix


def return_somethingv2():
    filename_categories = './data/sthsth2/category.txt'
    root_data = './video_frames/SSv2/jpg'
    filename_imglist_train = './data/sthsth2/ssv2_train.txt'
    filename_imglist_val = './data/sthsth2/ssv2_val.txt'
    filename_imglist_balance = './data/sthsth2/balance.txt'
    prefix = 'img_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, filename_imglist_balance, root_data, prefix


def return_kinetics():
    filename_categories = './data/kinetics/kinetics_label_map.txt'
    root_data = './video_frames/kinetics_frames'
    filename_imglist_train = './data/kinetics/train_videofolder.txt'
    filename_imglist_val = './data/kinetics/val_videofolder.txt'
    filename_imglist_balance = './data/kinetics/balance.txt'
    prefix = 'img_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, filename_imglist_balance, root_data, prefix


def return_activitynet():
    filename_categories = './data/Activitynet/lable/output.txt'
    root_data = './video_frames/Activitynet/train'
    filename_imglist_train = './data/Activitynet/lable/train_split1.txt'
    filename_imglist_val = './data/Activitynet/lable/val_split.txt'
    filename_imglist_balance = './data/Activitynet/lable/balance.txt'
    prefix = 'img_{:05d}.jpg'

    return (filename_categories, filename_imglist_train, filename_imglist_val,
            filename_imglist_balance, root_data, prefix)


def return_dataset(dataset):
    dict_single = {'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'ssv2': return_somethingv2, 'kinetics': return_kinetics,
                   'activitynet': return_activitynet}

    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, file_imglist_balance, root_data, prefix = dict_single[dataset]()
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_imglist_balance = os.path.join(ROOT_DATASET, file_imglist_balance)

    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, file_imglist_balance, root_data, prefix
