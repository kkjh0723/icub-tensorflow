clear all; clc; clf;


WIDTH = 64;
HEIGHT = 48;

trials = load('./trialList.txt');
figure(1);

for i = 1:size(trials,1)
    obj1 = trials(i,1);
    obj2 = trials(i,2);
    gesture = trials(i,3);
    
    data = load(sprintf('./vision_%04d_0000_0000_0000_0000_0000.txt',obj1));
    length = size(data,1);
    for idxStep = 1:4:length
        tempFrame = (transpose(reshape(data(idxStep,:),[64 48]))+1)./2;
        
        subplot(1,1,1);
        imshow(tempFrame,'InitialMagnification',600)
        title(sprintf('step: %03d',idxStep));
        drawnow
        
    end
end