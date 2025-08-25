function data = PrepareData(NNdata,pr)
indxOfTrainData = randperm(numel(NNdata.X),ceil(pr.trPercentage*numel(NNdata.X)));
indxOfTestData  = find(~ismember([1:numel(NNdata.X)],indxOfTrainData));
X = NNdata.X;
Y = NNdata.Y;
PtrainX = X(indxOfTrainData);
PtrainY = Y(indxOfTrainData);
data.TrainIn    = PtrainX;
data.TrainOut   = PtrainY;
PtestX = X(indxOfTestData);
PtestY = Y(indxOfTestData);
data.TestIn     = PtestX;
data.TestOut    = PtestY;
end