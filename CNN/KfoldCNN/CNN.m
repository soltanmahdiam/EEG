clear
close all
clc

load pop.mat

%% Classification
Features = {'Mean';'Variance';'Abolute Energy';'Auto Regression';'Shannon Entropy';'Hjorth Mobility'};

cn = 0;
for fi = 1:7

    if fi == 2
        continue
    end
    cn = cn + 1;

    indv.features = 0;
    indv.label = 0;
    Pop = repmat(indv,[numel(Pop2) 1]);

    for i=1:numel(Pop2)
        pp = Pop2(i).features;
        Pop(i).features = pp(fi,:)';
        Pop(i).label = Pop2(i).label;
    end

    for i = 1:numel(Pop)
        NNdata.X{i} = Pop(i).features;
        Y(i) = Pop(i).label;
    end
    NNdata.Y = categorical(Y);

    %% CNN
    k = 5;
    Indxs = crossvalind('Kfold', [Pop.label]', k);

    Xin = NNdata.X;
    You = NNdata.Y;

    for fol = 1:k

        disp(['CNN Training. Feature : ' Features{cn} '   fold : ' num2str(fol)])

        tstIndxs = (Indxs == fol);
        trnIndxs = ~tstIndxs;

        data.TrainIn    = Xin(find(trnIndxs));
        data.TrainOut   = You(find(trnIndxs));
        data.TestIn     = Xin(find(tstIndxs));
        data.TestOut    = You(find(tstIndxs));

        for i = 1:numel(data.TrainIn)
            IN = data.TrainIn{i}';
            XTrain(:,:,1,i) = IN;               %#ok 
            YTrain(i,1)     = data.TrainOut(i); %#ok 
        end

        for i = 1:numel(data.TestIn)
            IN = data.TestIn{i}';
            XTest(:,:,1,i) = IN;               %#ok  
            YTest(i,1)     = data.TestOut(i);  %#ok 
        end

        layers = [
            imageInputLayer([1 31 1])
            convolution2dLayer(3,8,'Padding','same')
            batchNormalizationLayer
            reluLayer
            convolution2dLayer(3,16,'Padding','same')
            batchNormalizationLayer
            reluLayer
            convolution2dLayer(3,32,'Padding','same')
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.3)
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer('Name','output')
            ];

        lgraph = layerGraph(layers);
        % plot(lgraph)

        miniBatchSize  = 128;
        options = trainingOptions('adam', ...
            'MiniBatchSize',miniBatchSize, ...
            'MaxEpochs',64, ...
            'InitialLearnRate',1e-3, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropFactor',0.1, ...
            'LearnRateDropPeriod',8, ...
            'Shuffle','every-epoch', ...
            'Plots','none', ...
            'Verbose',false);

        net = trainNetwork(XTrain,YTrain,layers,options);

        YPredicted = double(classify(net,XTest,'MiniBatchSize',miniBatchSize));

        confmat = confusionmat(YPredicted,double(YTest)); % where response is the last column in the dataset representing a class
        TP = confmat(2, 2);
        TN = confmat(1, 1);
        FP = confmat(1, 2);
        FN = confmat(2, 1);


        Acc(fol) = (TP + TN) / (TP + TN + FP + FN);  %# ok
        Sen(fol) = TP / (FN + TP);  %# ok
        Spec(fol) = TN / (TN + FP);  %# ok

        z = FP / (FP+TN);
        X = [0;Sen(fol);1];
        Y = [0;z;1];
        AC(fol) = trapz(Y,X);  %# ok  % This way is used for only binary classification

    end
    Accuracymean(cn,1)    = mean(Acc);   %# ok
    Accuracyvar(cn,1)     = var(Acc);    %# ok
    Sensitivitymean(cn,1) = mean(Sen);   %# ok
    Sensitivityvar(cn,1)  = var(Sen);    %# ok
    Specificitymean(cn,1) = mean(Spec);  %# ok
    Specificityvar(cn,1)  = var(Spec);   %# ok
    AUCmean(cn,1)         = mean(AC);    %# ok
    AUCvar(cn,1)          = var(AC);     %# ok

end
Accuracy    = table(Accuracymean*100,Accuracyvar*100,'VariableNames',{'Mean','Variance'});
Sensitivity = table(Sensitivitymean*100,Sensitivityvar*100,'VariableNames',{'Mean','Variance'});
Specificity = table(Specificitymean*100,Sensitivityvar*100,'VariableNames',{'Mean','Variance'});
AUC         = table(AUCmean*100,AUCvar*100,'VariableNames',{'Mean','Variance'});

Meters = table(Features,Accuracy,Sensitivity,Specificity,AUC);
T2 = inner2outer(Meters);
TableWriter(T2,'CNN_Time.xlsx')

%% Pop1
clear

load pop.mat

Features = {'CorrPearson';'coherence';'PLI';'PLV'};


cn = 0;
for fi = 1:4
   
    cn = cn + 1;

    indv.features = 0;
    indv.label = 0;
    Pop = repmat(indv,[numel(Pop1) 1]);
    
    for i=1:numel(Pop1)
        pp = Pop1(i).features;
        Pop(i).features = pp(:,fi);
        Pop(i).label = Pop1(i).label;
    end
    
    pr.trPercentage    = 0.8;
    
    for i = 1:numel(Pop)
        NNdata.X{i} = Pop(i).features;
        Y(i) = Pop(i).label;
    end
    NNdata.Y = categorical(Y);

     k = 5;
    Indxs = crossvalind('Kfold', [Pop.label]', k);

    Xin = NNdata.X;
    You = NNdata.Y;
    
    for fol = 1:k

        disp(['CNN Training. Feature : ' Features{cn} '   fold : ' num2str(fol)])
        
        tstIndxs = (Indxs == fol);
        trnIndxs = ~tstIndxs;

        data.TrainIn    = Xin(find(trnIndxs));
        data.TrainOut   = You(find(trnIndxs));
        data.TestIn     = Xin(find(tstIndxs));
        data.TestOut    = You(find(tstIndxs));

    for i = 1:numel(data.TrainIn)
        IN = data.TrainIn{i}';
        XTrain(:,:,1,i) = IN;                 %#ok 
        YTrain(i,1)     = data.TrainOut(i);   %#ok 
    end
    
    for i = 1:numel(data.TestIn)
        IN = data.TestIn{i}';
        XTest(:,:,1,i) = IN;                  %#ok 
        YTest(i,1)     = data.TestOut(i);     %#ok 
    end

    layers = [
        imageInputLayer([1 465 1])
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.3)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer('Name','output')
        ];
    
    lgraph = layerGraph(layers);
    plot(lgraph)
    
    miniBatchSize  = 128;
  %  validationFrequency = floor(numel(data.TrainOut')/miniBatchSize);
    options = trainingOptions('adam', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',64, ...
        'InitialLearnRate',1e-3, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',8, ...
        'Shuffle','every-epoch', ...
        'Plots','none', ...
        'Verbose',false);
    
    net = trainNetwork(XTrain,YTrain,layers,options);
    
    YPredicted = double(classify(net,XTest,'MiniBatchSize',miniBatchSize));
   
    confmat = confusionmat(YPredicted,double(YTest)); % where response is the last column in the dataset representing a class
    TP = confmat(2, 2);
        TN = confmat(1, 1);
        FP = confmat(1, 2);
        FN = confmat(2, 1);
        
        
        Acc(fol) = (TP + TN) / (TP + TN + FP + FN);  %# ok
        Sen(fol) = TP / (FN + TP);  %# ok
        Spec(fol) = TN / (TN + FP);  %# ok
        
        z = FP / (FP+TN);
        X = [0;Sen(fol);1];
        Y = [0;z;1];
        AC(fol) = trapz(Y,X);  %# ok  % This way is used for only binary classification
    end
    Accuracymean(cn,1)    = mean(Acc);   %# ok
    Accuracyvar(cn,1)     = var(Acc);    %# ok
    Sensitivitymean(cn,1) = mean(Sen);   %# ok
    Sensitivityvar(cn,1)  = var(Sen);    %# ok
    Specificitymean(cn,1) = mean(Spec);  %# ok
    Specificityvar(cn,1)  = var(Spec);   %# ok
    AUCmean(cn,1)         = mean(AC);    %# ok
    AUCvar(cn,1)          = var(AC);     %# ok
    
end
Accuracy    = table(Accuracymean*100,Accuracyvar*100,'VariableNames',{'Mean','Variance'});
Sensitivity = table(Sensitivitymean*100,Sensitivityvar*100,'VariableNames',{'Mean','Variance'});
Specificity = table(Specificitymean*100,Sensitivityvar*100,'VariableNames',{'Mean','Variance'});
AUC         = table(AUCmean*100,AUCvar*100,'VariableNames',{'Mean','Variance'});

Meters = table(Features,Accuracy,Sensitivity,Specificity,AUC);
T2 = inner2outer(Meters);
TableWriter(T2,'CNN_App.xlsx')