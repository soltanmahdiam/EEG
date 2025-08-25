clear
close all
clc

load pop2.mat

for i = 1:numel(Pop2)
    temp = Pop2(i).features;
    temp([1 5],:) = [];
    Pop2(i).features = temp;
    clear temp
end

for i = 1:numel(Pop1)
    temp = Pop1(i).features;
    temp(:,2) = [];
    Pop1(i).features = temp;
    clear temp
end


%% Data Preparation
for i = 1:numel(Pop1)
    NNdata.X{i} = Pop1(i).features;
    NNdata2.X{i} = Pop2(i).features;
    Y(i) = Pop1(i).label; %# ok
end
NNdata.Y = categorical(Y);

pr.trPercentage = 0.8;
data = PrepareData(NNdata,pr);

for i = 1:numel(data.TrainIn)
    IN = data.TrainIn{i}';
    XTrain(:,:,1,i) = IN; %#ok 
    YTrain(i,1)     = data.TrainOut(i); %#ok 
end

for i = 1:numel(data.TestIn)
    IN = data.TestIn{i}';
    XValidation(:,:,1,i) = IN; %#ok 
    YValidation(i,1)     = data.TestOut(i); %#ok 
end


% pop 2
NNdata2.Y = categorical(Y);
dataPop2 = PrepareData(NNdata2,pr);


for i = 1:numel(dataPop2.TrainIn)

    Xp{i,1} = dataPop2.TrainIn{i}'; %# ok

end


dataPop2.TrainIn = Xp;


X1Train = XTrain;
X2Train = dataPop2.TrainIn;
TTrain  = YTrain;

dsX1Train = arrayDatastore(X1Train,IterationDimension=4);
dsX2Train = arrayDatastore(X2Train,"OutputType","same");
dsTTrain = arrayDatastore(TTrain);
dsTrain = combine(dsX1Train,dsX2Train,dsTTrain);


%% Neural NetWork

[h,w,numChannels,numObservations] = size(X1Train);
numFeatures = 31;
numClasses = numel(categories(TTrain));

imageInputSize = [h w numChannels];
filterSize = 3;
numFilters = 8;

hiddenSize = 12;




layers = [
    imageInputLayer(imageInputSize)

    convolution2dLayer(filterSize,numFilters,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    flattenLayer
    concatenationLayer(1,2,Name="cat")

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

lgraph = layerGraph(layers);

layers = [
    sequenceInputLayer(numFeatures)
    bilstmLayer(hiddenSize)
    batchNormalizationLayer
    reluLayer
    bilstmLayer(hiddenSize/2,OutputMode="last",Name="lstm")
    ];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,"lstm","cat/in2");

figure
plot(lgraph)

miniBatchSize = 64;

options = trainingOptions('adam', ... % 'sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',70, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);


net = trainNetwork(dsTrain,lgraph,options);


%% Verifing

for i = 1:numel(dataPop2.TestIn)

    XpTest{i,1} = dataPop2.TestIn{i}'; %# ok

end


dataPop2.TestIn = Xp;


X1Test = XValidation;
X2Test = XpTest;
TTest  = YValidation;

dsX1Test = arrayDatastore(X1Test,IterationDimension=4);
dsX2Test = arrayDatastore(X2Test,'OutputType','same');
dsTTest  = arrayDatastore(TTest);
dsTest   = combine(dsX1Test,dsX2Test);


YPredicted = double(classify(net,dsTest,'MiniBatchSize',miniBatchSize));

%% 

confmat = confusionmat(YPredicted,double(YValidation)); % where response is the last column in the dataset representing a class
TP = confmat(2, 2);
TN = confmat(1, 1);
FP = confmat(1, 2);
FN = confmat(2, 1);


Accuracy = (TP + TN) / (TP + TN + FP + FN);
Sensitivity = TP / (FN + TP);
specificity = TN / (TN + FP);


z = FP / (FP+TN);
X = [0;Sensitivity;1];
Y = [0;z;1];
AUC = trapz(Y,X);  % This way is used for only binary classification
disp(['Accuracy: ' num2str(Accuracy*100) '%   Sensitivity: ' num2str(Sensitivity*100) ...
    '%   Specificity: ' num2str(specificity*100) '%   AUC: ' num2str(AUC*100) '%'])
