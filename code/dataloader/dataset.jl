using HDF5;

function read_dataset(data_path::String, train::Bool)
    
    # Read train/test dataset
    if train
        split = "train"
    else
        split = "test"
    end
    folders = open("$data_path/2018LA_Seg_Training Set/$split.list", "r") do file
        return readlines(file)
    end

    # Read all h5 images & masks
    images = []
    labels = []
    for folder in folders
        img, label = h5open("$data_path/2018LA_Seg_Training Set/$folder/mri_norm2.h5", "r") do file
            read(file, "image"), read(file, "label")
        end
        push!(images, img)
        push!(labels, label)
    end

    return images, labels
end




