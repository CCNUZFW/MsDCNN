function net = DnCNN_Init()
net = dagnn.DagNN();

% conv + relu\
blockNum = 1;
inVar = 'input';
channel= 1; % grayscale image

dilate = [1,1];
dims   = [32,32,1,10];
pad    = [0,0];
stride = [32,32];
lr     = [1,1];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr)
% [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
 
% % dims   = [1,1,256,1024];
% % pad    = [0,0];
% % stride = [1,1];
% % lr     = [1,0];
% % [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
% % % [net, inVar, blockNum] = addReLU(net, blockNum, inVar);
% % 
% % [net, inVar, blockNum] = addbcs_init_rec_dag(net, blockNum, inVar);
% 
% 
dims     = [32,32,1,10];
crop     = [0,0];
upsample = [32,32];
lr       = [1,1];
[net, inVar, blockNum] = addConvt(net, blockNum, inVar, dims, crop, upsample, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);


inVar = 'relu3';
dilate = [1,1];
dims   = [3,3,1,32];
pad    = [1,1];
stride = [1,1];
lr     = [1,1];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

for i = 1:3
    dims   = [3,3,32,32];
    dilate = [1,1];
    pad    = [1,1];
    stride = [1,1];
    lr     = [1,0];
    [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr);
    [net, inVar, blockNum] = addReLU(net, blockNum, inVar);
end

% % inVar = 'reshape_wzhshi3';
inVar = 'relu3';
dilate = [2,2];
dims   = [3,3,1,32];
pad    = [2,2];
stride = [1,1];
lr     = [1,1];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

dilate = [1,1];
dims   = [5,5,32,32];
pad    = [2,2];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

dilate = [2,2];
dims   = [3,3,32,32];
pad    = [2,2];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

dilate = [1,1];
dims   = [5,5,32,32];
pad    = [2,2];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
% % inVar = 'reshape_wzhshi3';

inVar = 'relu3';
dilate = [3,3];
dims   = [3,3,1,32];
pad    = [3,3];
stride = [1,1];
lr     = [1,1];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

dilate = [1,1];
dims   = [7,7,32,32];
pad    = [3,3];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

dilate = [3,3];
dims   = [3,3,32,32];
pad    = [3,3];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

dilate = [1,1];
dims   = [7,7,32,32];
pad    = [3,3];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

% 
% % dims   = [7,7,1,32];
% % pad    = [3,3];
% % stride = [1,1];
% % lr     = [1,1];
% % [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
% % [net, inVar, blockNum] = addReLU(net, blockNum, inVar);
% % 
% % for i = 1:3
% %     % conv + bn + relu
% %     dims   = [7,7,32,32];
% %     pad    = [3,3]; 
% %     stride = [1,1];
% %     lr     = [1,0];
% %     [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
% %     [net, inVar, blockNum] = addReLU(net, blockNum, inVar);
% %  end
[net, inVar, blockNum] = addConcat(net, blockNum, inVar);
% 
dims   = [3,3,96,96];
dilate = [1,1];
pad    = [1,1];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, dilate, pad, stride,lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);




dims   = [3,3,96,channel];
dilate = [1,1];
pad    = [1,1];
stride = [1,1];
lr     = [1,0]; % or [1,1], it does not influence the results
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, dilate, pad, stride,lr);
% [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);

% sum
% inVar = {inVar,'input'};
% [net, inVar, blockNum] = addSum(net, blockNum, inVar);
outputName = 'prediction';
net.renameVar(inVar,outputName)

% loss
net.addLayer('loss', dagnn.Loss('loss','L2'), {'prediction','label'}, {'objective'},{});
net.vars(net.getVarIndex('prediction')).precious = 1;
end



% Add a Concat layer
function [net, inVar, blockNum] = addConcat(net, blockNum, inVar)

outVar   = sprintf('concat%d', blockNum);
layerCur = sprintf('concat%d', blockNum);
inVar = dagnn.Concat('dim',3);
net.addLayer(layerCur,inVar,{'relu11', 'relu19', 'relu27'}, {outVar},{});
inVar = outVar;
blockNum = blockNum + 1;
end

% Add a loss layer
function [net, inVar, blockNum] = addLoss(net, blockNum, inVar)

outVar   = 'objective';
layerCur = sprintf('loss%d', blockNum);

block    = dagnn.Loss('loss','L2');
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a sum layer
function [net, inVar, blockNum] = addSum(net, blockNum, inVar)

outVar   = sprintf('sum%d', blockNum);
layerCur = sprintf('sum%d', blockNum);

block    = dagnn.Sum();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a relu layer
function [net, inVar, blockNum] = addReLU(net, blockNum, inVar)

outVar   = sprintf('relu%d', blockNum);
layerCur = sprintf('relu%d', blockNum);

block    = dagnn.ReLU('leak',0);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end

function [net, inVar, blockNum] =addbcs_init_rec_dag(net, blockNum, inVar)

outVar   = sprintf('reshape_wzhshi%d', blockNum);
layerCur = sprintf('reshape_wzhshi%d', blockNum);

block    = dagnn.bcs_init_rec_dag('dims',[32,32]);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end

% Add a bnorm layer
function [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch)

trainMethod = 'adam';
outVar   = sprintf('bnorm%d', blockNum);
layerCur = sprintf('bnorm%d', blockNum);

params={[layerCur '_g'], [layerCur '_b'], [layerCur '_m']};
net.addLayer(layerCur, dagnn.BatchNorm('numChannels', n_ch), {inVar}, {outVar},params) ;

pidx = net.getParamIndex({[layerCur '_g'], [layerCur '_b'], [layerCur '_m']});
b_min                           = 0.025;
net.params(pidx(1)).value       = clipping(sqrt(2/(9*n_ch))*randn(n_ch,1,'single'),b_min);
net.params(pidx(1)).learningRate= 1;
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(1)).trainMethod = trainMethod;

net.params(pidx(2)).value       = zeros(n_ch, 1, 'single');
net.params(pidx(2)).learningRate= 1;
net.params(pidx(2)).weightDecay = 0;
net.params(pidx(2)).trainMethod = trainMethod;

net.params(pidx(3)).value       = [zeros(n_ch,1,'single'), 0.01*ones(n_ch,1,'single')];
net.params(pidx(3)).learningRate= 1;
net.params(pidx(3)).weightDecay = 0;
net.params(pidx(3)).trainMethod = 'average';

inVar    = outVar;
blockNum = blockNum + 1;
end


% add a ConvTranspose layer
function [net, inVar, blockNum] = addConvt(net, blockNum, inVar, dims, crop, upsample, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('convt%d', blockNum);

layerCur    = sprintf('convt%d', blockNum);

convBlock = dagnn.ConvTranspose('size', dims, 'crop', crop,'upsample', upsample, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f  = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*dims(4))) ; %improved Xavier
net.params(f).value        = sc*randn(dims, 'single');
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(3), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end


% add a Conv layer
function [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts    = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('conv%d', blockNum);
layerCur    = sprintf('conv%d', blockNum);

convBlock   = dagnn.Conv('size', dims,'pad', pad,'stride', stride, 'dilate',dilate,...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*max(dims(3), dims(4)))); %improved Xavier
net.params(f).value        = sc*randn(dims, 'single') ;
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(4), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end



function A = clipping(A,b)
A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;
end

