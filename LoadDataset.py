import tonic
import torch
import torch.nn as nn
import tonic.datasets
import os



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATASET_CONFIGS = {
    "DVSGesture": {
        "class": tonic.datasets.DVSGesture,
        "sensor_size": tonic.datasets.DVSGesture.sensor_size,
        "num_classes": 11,
        "has_train_test_split": True,
    },
    "ASLDVS": {
        "class": tonic.datasets.ASLDVS,
        "sensor_size": (240, 180, 2),
        "num_classes": 24,
        "has_train_test_split": False,
    },
    "SHD": {
        "class": tonic.datasets.hsd.SHD,
        "sensor_size": (700, 1, 2),
        "num_classes": 20,
        "has_train_test_split": True,
    },
}


def load_dataset(dataset_name, dataset_path, w=32, h=32, n_frames=32):

    
    # Get config for selected dataset
    config = DATASET_CONFIGS[dataset_name]
    dataset_class = config["class"]
    sensor_size = config["sensor_size"]
    num_classes = config["num_classes"]
    has_train_test_split = config["has_train_test_split"]



    if dataset_name == "SHD":
        cache_root = f"{dataset_path}/{dataset_name.lower()}/700x1_T{n_frames}"
    else:
        cache_root = f"{dataset_path}/{dataset_name.lower()}/{w}x{h}_T{n_frames}"

    loadCacheOnly = cache_exists = os.path.exists(f"{cache_root}/train") and os.path.exists(f"{cache_root}/test")






    if dataset_name == "SHD":
        transforms = tonic.transforms.Compose([
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_frames),
        ])
    else:
        transforms = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.Downsample(sensor_size=sensor_size, target_size=(w, h)),
            tonic.transforms.ToFrame(sensor_size=(w, h, 2), n_time_bins=n_frames),
        ])
    

    # Download and load dataset (Tonic automatically checks if already downloaded)

    train_dataset = None
    test_dataset = None
    if (not loadCacheOnly):
        if has_train_test_split:
            # DVSGesture has official train/test split
            train_dataset = dataset_class(save_to=dataset_path, transform=transforms, train=True)
            test_dataset = dataset_class(save_to=dataset_path, transform=transforms, train=False)
        else:
            # ASLDVS needs manual split
            full_dataset = dataset_class(save_to=dataset_path, transform=transforms)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
            )
    

    # Cache the preprocessed data



    
        
    cached_train = tonic.DiskCachedDataset(train_dataset, cache_path=f"{cache_root}/train")
    cached_test = tonic.DiskCachedDataset(test_dataset, cache_path=f"{cache_root}/test")

    if (not loadCacheOnly):
        for _ in cached_train:
            pass
        for _ in cached_test:
            pass
    
    # print(f"Dataset: {dataset_name}")
    # print(f"Train samples: {len(cached_train)}")
    # print(f"Test samples: {len(cached_test)}")
    # print(f"Number of classes: {num_classes}")
    
    return cached_train, cached_test, num_classes





# The below code is to test out datasets, and to download/cache them. 

if __name__ == "__main__":
    dataset_name = "SHD"
    dataset_path = "/home/gauravgupta/CMPM118/data"  # change path if needed
    w, h, n_frames = 700, 1, 100 # typical temporal bin count for SHD

    cached_train, cached_test, num_classes = load_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        w=w,
        h=h,
        n_frames=n_frames,
    )

    sample, label = cached_train[0]
    print(sample.shape, label)

    import matplotlib.pyplot as plt
    plt.imshow(sample.sum(1).squeeze().T, aspect='auto')
    plt.title(f"Label: {label}")
    plt.xlabel("Time bins")
    plt.ylabel("Frequency channels")
    plt.show()


















# print("URL:", tonic.datasets.ASLDVS.url)
# print("Filename:", tonic.datasets.ASLDVS.filename)
# print("MD5:", tonic.datasets.ASLDVS.file_md5)

# dataset_name = "DVSGesture"  # or "DVSASL", "NMNIST", etc.
# dataset_path = '/home/gauravgupta/CMPM118/data'
# w,h=32,32
# n_frames=32


# transforms = tonic.transforms.Compose([
#     tonic.transforms.Denoise(filter_time=10000), # removes outlier events with inactive surrounding pixels for 10ms
#     tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(w,h)), # downsampling image
#     tonic.transforms.ToFrame(sensor_size=(w,h,2), n_time_bins=n_frames), # n_frames frames per trail
# ])

# train2 = tonic.datasets.DVSGesture(save_to=dataset_path, transform=transforms, train=True)
# test2 = tonic.datasets.DVSGesture(save_to=dataset_path, transform=transforms, train=False)
