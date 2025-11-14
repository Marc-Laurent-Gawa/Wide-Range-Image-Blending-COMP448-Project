%% COMMENTS
%Before running the section
%It is assumed the setup steps listed in the read me word file
%have been followed, otherwise the script will not work properly

%% RUNNING THE MODEL ON  THE SPLIT DATASET

pyrunfilestatus = system('python test_alternate.py', '-echo');

if pyrunfilestatus == 0
    disp("Python script completed successfully.");
else
    disp("Python script encountered an error.");
end


%% VIEW THE RESULTS
batchSize = str2double(inputdlg("Enter batch size:", "Batch Size", 1, {"5"}));

if isnan(batchSize) || batchSize <= 0
    error("Invalid batch size entered.");
end

imageFolder = fullfile(pwd, "results", "result1-2");

imageFiles = dir(fullfile(imageFolder, "*.png"));

if isempty(imageFiles)
    error("No images found in the folder.");
end

numImages = length(imageFiles);
disp("Starting batch processing...");

for i = 1:batchSize:numImages

    if i > 1
        userInput = input("Press ENTER to continue to the next batch, or type 'stop' to end: ","s");
        if strcmpi(userInput, 'stop')
            disp("Processing stopped by user.");
            return;
        end
    end

    batchEnd = min(i + batchSize - 1, numImages);

    fprintf("\nProcessing batch %d to %d...\n", i, batchEnd);

    for j = i:batchEnd
        imgPath = fullfile(imageFolder, imageFiles(j).name);
        fprintf("  Processing file: %s\n", imageFiles(j).name);
        img = imread(imgPath);
        figure;   
        imshow(img);
    end

    disp("Batch complete.");
end

disp("All batches finished.");