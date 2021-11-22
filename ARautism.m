clc;clear;
close all;
%% Data Reading
bodyc=imread('bodyc.jpg');
bodyd=imread('bodyd.jpg');
facec=imread('facec.jpg');
faced=imread('faced.jpg');
gesc=imread('gesc.jpg');
gesd=imread('gesd.jpg');

% K-means Segmentation
L = imsegkmeans(bodyd,3);
B = labeloverlay(bodyd,L);

% Body Detection
peopleDetector = vision.PeopleDetector;
[bboxes,scores] = peopleDetector(B);
I1 = insertObjectAnnotation(B,'rectangle',bboxes,scores,'LineWidth',7,'FontSize',45,'Color',{'black'});
subplot(1,3,1);
subimage(bodyd);title('Depth Raw');
subplot(1,3,2);
subimage(B);title('K-Means Clustering');
subplot(1,3,3);
subimage(I1);title('Human Detection');
% Hint
disp(['Check if subject is in 2.5 meter distance from sensor']);
disp(['Yes = Stop   ,   No = Please got to the specific distance']);

% Face detection
faceDetector = vision.CascadeObjectDetector;
bboxes1 = faceDetector(facec);
IFaces = insertObjectAnnotation(facec,'rectangle',bboxes1(2,:),'Face','LineWidth',4,'FontSize',30,'Color',{'magenta'});   
figure;
cropedf=imcrop(facec,bboxes1(2,:)); 
subplot(1,3,1);
subimage(facec);title('Color Raw');
subplot(1,3,2);
subimage(IFaces);title('Viola Jones Face Detection');
subplot(1,3,3);
subimage(cropedf);title('Cropped Face');

% CNN Face Recognition
deepDatasetPath = fullfile('testfolder');
imds = imageDatastore(deepDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% Number of training (less than number of each class)
numTrainFiles = 15;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
layers = [
    % Input image size for instance: 512 512 3
    imageInputLayer([512 512 3])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    % Number of classes
        fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
disp(['CNN Face Recognition Accuracy Is = ']);
accuracy = sum(YPred == YValidation)/numel(YValidation)

% Extracting SURF features
imset = imageSet('gesture','recursive'); 
bag = bagOfFeatures(imset,'VocabularySize',10,'PointSelection','Detector');
surf = encode(bag,imset);
sizefinal=size(surf);
sizefinal=sizefinal(1,2);
surf(1:20,sizefinal+1)=1;
surf(21:40,sizefinal+1)=2;
dataknn=surf(:,1:10);
lblknn=surf(:,end);

% KNN classification
Mdl = fitcknn(dataknn,lblknn,'NumNeighbors',5,'Standardize',1)
rng(1); % For reproducibility
knndat = crossval(Mdl);
classError = kfoldLoss(knndat)
disp(['K-NN Gesture Classification Accuracy :   ' num2str(100-classError) ]);
% Hint
disp(['Check if the gesture correct ...']);
disp(['Yes = Start AR   ,   No = Please use proper gesture']);

%% Augmented Reality Part
vid = videoinput('kinect', 1);
set(vid,'FramesPerTrigger',1);
set(vid,'TriggerRepeat',Inf);
triggerconfig(vid, 'Manual');
vid.ReturnedColorspace = 'rgb';
start(vid);
spec=imread('owl.jpg');
for i=1:1:30        
        % video on every loop
        trigger(vid);
        % acquire the data 
        IM = (getdata(vid,1,'uint8'));
        % adjusting window
        IM_r=imcrop(IM,[255 254 600 600]);
        faceDetector = vision.CascadeObjectDetector; 
        % Detect faces and create a bounding box 
        bbox = step(faceDetector, IM_r); 
        TF=isempty(bbox);
        if(TF==0)
            st_x=bbox(1,1)-10;   
            st_y=bbox(1,2)-10;
            t_x=st_x+7;
            t_y=st_y+45;
        % Augmentation 
            for in=1:1:128
                for jn=1:1:128
                    if spec(in,jn,:)<20
                      IM_r(st_y,st_x,:)=spec(in,jn,:);
                    end
                             if spec(in,jn,:)<253
                      IM_r(st_y,st_x,:)=IM_r(st_y,st_x,:)+spec(in,jn,:);
                             end                   
                    st_x=st_x+1;
                    if(st_x==t_x+128)
                        st_x=t_x;
                        st_y=st_y+1;
                    end
                end
            end        
        end
% LIVE Video Stream Result
imshow(IM_r), title('Please Keep Your Distance'); 
end
    stop(vid);
    delete(vid);
disp(['Rehabilitation Estimation Step by Expert ...']);
