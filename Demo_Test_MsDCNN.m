%% testing set
addpath(fullfile('utilities'));

folderModel = 'model';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'Set11'}; % testing datasets
setTestCur  = imageSets{1};      % current testing dataset


showresult  = 1;
gpu         = 1;


% noiseSigma  = 25;

% load model
epoch       =0.1;

modelName   = 'DnCNN';

% case one: for the model in 'data/model'
%load(fullfile('data',folderModel,[modelName,'-epoch-',num2str(epoch),'.mat']));

% case two: for the model in 'utilities'
load(fullfile('utilities',[modelName,'-epoch-',num2str(epoch),'.mat']));

net = dagnn.DagNN.loadobj(net) ;

net.removeLayer('loss') ;
out1 = net.getVarIndex('prediction') ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;

net.mode = 'test';

if gpu
    net.move('gpu');
end

% read images
ext         =  {'*.jpg','*.png','*.bmp','*.pgm','*.tif'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
end

folderResultCur       =  fullfile(folderResult,[setTestCur]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end


% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));


for i = 1 : length(filePaths)
%      tic
    % read image
    image = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
     if size(image,3) == 3
         image = modcrop(image,32); 
         image = rgb2ycbcr(image); 
         image = image(:,:,1);
     end
    [w,h,c]=size(image);
    if c==3
        image = rgb2gray(image);
    end
    
    label =im2single(image);
   
    input = label;
    if gpu
        input = gpuArray(input);
     end
    net.eval({'input', input}) ;
    % output (single)
   output = gather(squeeze(gather(net.vars(out1).value)));

    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    if showresult
       % imshow(cat(2,im2uint8(label),im2uint8(output)));
       imshow(cat(2,im2uint8(output)));
       % title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        drawnow;
       % pause()
    end
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
end


disp([mean(PSNRs),mean(SSIMs)]);




