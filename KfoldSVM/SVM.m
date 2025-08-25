clear
close all
clc

load ..\Data\pop.mat

%% Classification
Features = {'Variance';'Abolute Energy';'Auto Regression';'Shannon Entropy';'Hjorth Mobility'};
 
cn = 0;
for fi = 1:7

    if fi==1 || fi ==2
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
    
%     pr.trPercentage    = 0.8;

    k = 5;
    Indxs = crossvalind('Kfold', [Pop.label]', k);
    
    for fol = 1:k
        
        tstIndxs = (Indxs == fol);
        trnIndxs = ~tstIndxs;
        SVMModel = fitcsvm([Pop(trnIndxs).features]',[Pop(trnIndxs).label]','KernelScale','auto','Standardize',true,...
        'OutlierFraction',0.05);
        SVMdata.SVMOut = predict(SVMModel,[Pop(tstIndxs).features]');
        

        % Calc Error
        
        confmat = confusionmat(SVMdata.SVMOut,[Pop(tstIndxs).label]'); % where response is the last column in the dataset representing a class
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
TableWriter(T2,'SVM_Time.xlsx')
%% Pop1

Features = {'CorrPearson';'coherence';'PLI';'PLV'};
clear Accuracy Specificity Sensitivity AUC Accuracymean Accuracyvar Sensitivitymean Sensitivityvar Specificitymean
clear Specificityvar AUCmean AUCvar Acc Spec AC

cn = 0;
for fi = 1:4
   
    cn = cn + 1;

    indv.features = 0;
    indv.label = 0;
    Pop = repmat(indv,[numel(Pop2) 1]);
    
    for i=1:numel(Pop1)
        pp = Pop1(i).features;
        Pop(i).features = pp(:,fi);
        Pop(i).label = Pop1(i).label;
    end
    
    
    Indxs = crossvalind('Kfold', [Pop.label]', k);
    
    for fol = 1:k
        
        tstIndxs = (Indxs == fol);
        trnIndxs = ~tstIndxs;
        SVMModel = fitcsvm([Pop(trnIndxs).features]',[Pop(trnIndxs).label]','KernelScale','auto','Standardize',true,...
        'OutlierFraction',0.05);
        SVMdata.SVMOut = predict(SVMModel,[Pop(tstIndxs).features]');
        

        % Calc Error
        
        confmat = confusionmat(SVMdata.SVMOut,[Pop(tstIndxs).label]'); % where response is the last column in the dataset representing a class
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
TableWriter(T2,'SVM_App.xlsx')


