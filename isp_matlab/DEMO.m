clc
clear
metadataFilename = '0001_METADATA_RAW_010.MAT';
metadata = load(fullfile(metadataFilename));
metadata = metadata.metadata;

data_dir = '..\\dataset\\wb_scene_clean_postprocessed\\bear\\'
filepaths = dir(fullfile(data_dir, '*.tiff'))
for k = 1:length(filepaths)
    imNormRaw = imread(fullfile(data_dir, filepaths(k).name));
    imSrgb2 = run_pipeline(imNormRaw, metadata, 'raw', 'tone');
    base_name = filepaths(k).name(1:end-5);
    imwrite(imSrgb2, [base_name,'_sRGB.png']);
end

