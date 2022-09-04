rng('default')

addpath('utilities');
%-------------------------------------------------------------------------
% Configuration
%-------------------------------------------------------------------------
opts.modelName        = 'DnCNN'; % model name
opts.learningRate     = [logspace(-3,-3,50) logspace(-4,-4,30) logspace(-5,-5,20)];% you can change the learning rate
opts.batchSize        = 64; % 
opts.gpus             =[1]; 
opts.numSubBatches    = 2;

% solver
opts.solver           = 'Adam'; % global
opts.derOutputs       = {'objective',1} ;

opts.backPropDepth    = Inf;
%-------------------------------------------------------------------------
%   Initialize model
%-------------------------------------------------------------------------

net  = feval([opts.modelName,'_Init']);

%-------------------------------------------------------------------------
%   Train
%-------------------------------------------------------------------------

[net, info] = DnCNN_train_dag(net,  ...
    'learningRate',opts.learningRate, ...
    'derOutputs',opts.derOutputs, ...
    'numSubBatches',opts.numSubBatches, ...
    'backPropDepth',opts.backPropDepth, ...
    'solver',opts.solver, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;






