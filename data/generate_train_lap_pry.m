clear;
close all;
folder = '/home/datasets/race/all_256'

savepath = 'train.h5';

size_label = 256;
scale = 4;
size_input = size_label/scale;
size_x2 = size_label/2;
stride = 256;
% downsizing
downsizes = [1];

data = zeros(size_input, size_input, 3, 1);
label_x2 = zeros(size_x2, size_x2, 3, 1);
label_x4 = zeros(size_label, size_label, 3, 1);
count = 0;
margain = 0;

% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.jpeg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

length(filepaths)

filepaths = filepaths(randperm(length(filepaths)))
chunksz = 64;
created_flag = false;
totalct = 0;

for i = 1 : length(filepaths)
    fprintf('%d: %s\n', i, filepaths(i).name)
    for downsize = 1 : length(downsizes)
        image = imread(fullfile(folder,filepaths(i).name));

        if size(image,3)==3
            % image = rgb2ycbcr(image);
            im_label = image;
            [hei,wid,channels] = size(im_label);

            for x = 1 + margain : stride : hei-size_label+1 - margain
                for y = 1 + margain :stride : wid-size_label+1 - margain
                    subim_label = im_label;
                    subim_label_x2 = imresize(subim_label,1/scale*2,'bicubic');
                    subim_input = imresize(subim_label,1/scale,'bicubic');

                    count=count+1;
                    batchdata(:, :, :, count) = subim_input;
                    batchlabs_x2(:, :, :, count) = subim_label_x2;
                    batchlabs(:, :, :, count) = subim_label;

                    if count == chunksz
                        startloc = struct('dat',[1,1,1,totalct+1], 'lab_x2', [1,1,1,totalct+1], 'lab_x4', [1,1,1,totalct+1]);
                        curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs_x2, batchlabs, ~created_flag, startloc, chunksz);
                        created_flag = true;
                        totalct = curr_dat_sz(end)
                        count = 0;
                    end
                end
            end
        end
    end
end

h5disp(savepath);
