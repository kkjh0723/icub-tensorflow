clear all; close all; clc;

dirName = './';   %# folder path
%trialInfo = load('./trialList.txt');
trialInfo = load('./trialList.txt');
num_of_result = size(trialInfo,1);

targetFileName = './0_result_objPosition_stat.txt';
fid=fopen(targetFileName,'w');

result = zeros(num_of_result,1);
numS = 0;
numF = 0;
numS2 = 0;
numF2 = 0;
numSall = 0;
numFall = 0;

numConfusion = 0;
numS_pnt_left = 0;
numS_pnt_right = 0;
numS_obj_tall = 0;
numS_obj_long = 0;
numS_left_tall = 0; %
numS_left_long = 0; %
numS_right_tall = 0; %
numS_right_long = 0; %


totalLeft = 0;
totalRight = 0;
totalBig = 0;
totalSmall = 0;
totalCircle = 0;
totalBox = 0;

sLeft = 0;
sRight = 0;
sBig = 0;
sSmall = 0;
sCircle = 0;
sBox = 0;

for idxTrial = 1:num_of_result
    trialIDNUM = trialInfo(idxTrial,1);
    trialIDNUM2 = trialInfo(idxTrial,2);
    trialIDNUM3 = trialInfo(idxTrial,3);
    trialIDNUM4 = trialInfo(idxTrial,4);
    trialIDNUM5 = trialInfo(idxTrial,5);
    trialIDNUM6 = trialInfo(idxTrial,6);
    trialSeqNum = trialInfo(idxTrial,7);
    
    position = load(sprintf('./temp/outputObj_%04d_%04d_%04d_%04d_%04d_%04d.txt', ...
        trialIDNUM,trialIDNUM2,trialIDNUM3,trialIDNUM4,trialIDNUM5,trialIDNUM6));
    
    fprintf(fid,'%04d\t%04d\t%04d\t', ...,
        trialIDNUM,trialIDNUM2,trialIDNUM3);
    
    fprintf(fid,'%04d\t',trialSeqNum);
    
    
    rndObjIdentifier = (trialIDNUM - mod(trialIDNUM,1000) )/ 1000;
    
    if(rndObjIdentifier > 3)
        %         objX = (mod(trialIDNUM,1000) - mod(trialIDNUM,100) )/ 100;
        %         objType = (mod(trialIDNUM,100) - mod(trialIDNUM,10) )/ 10;
        %         objRot = mod(trialIDNUM,10);
    else
        objType = (trialIDNUM - mod(trialIDNUM,1000) )/ 1000;
        objSize = (mod(trialIDNUM,1000) - mod(trialIDNUM,100) )/ 100;
        objLoc = (mod(trialIDNUM,100) - mod(trialIDNUM,10) )/ 10;
        objRot = mod(trialIDNUM,10); % Not used..
    end
    
    objType = (trialIDNUM - mod(trialIDNUM,1000) )/ 1000;
    objSize = (mod(trialIDNUM,1000) - mod(trialIDNUM,100) )/ 100;
    objLoc = (mod(trialIDNUM,100) - mod(trialIDNUM,10) )/ 10;
    objType = objType - 3;
    
    
    if(objType == 1)
        fprintf(fid,'BAL\t');
    elseif(objType == 2)
        fprintf(fid,'BOX\t');
    elseif(objType == 3)
        fprintf(fid,'CYL\t');
    end
    
    if(objSize == 1)
        fprintf(fid,'BIGG\t');
    elseif(objSize == 2)
        fprintf(fid,'SMAL\t');
    end
    
    if(objLoc <= 4)
        fprintf(fid,'LEFT\t');
    else
        fprintf(fid,'RGHT\t');
    end
    
    
    otherType = (trialIDNUM2 - mod(trialIDNUM2,1000) )/ 1000;
    otherSize = (mod(trialIDNUM2,1000) - mod(trialIDNUM2,100) )/ 100;
    otherLoc = (mod(trialIDNUM2,100) - mod(trialIDNUM2,10) )/ 10;
    otherRot = mod(trialIDNUM2,10);
    
    
    gesPerson = (trialIDNUM3 - mod(trialIDNUM3,100) )/ 100;
    gesType = (mod(trialIDNUM3,100) - mod(trialIDNUM3,10) )/ 10;
    gesTrial = mod(trialIDNUM3,10);
    
    
    if(gesType == 8)%1)
        fprintf(fid,'G.RGHT\t');
        totalRight = totalRight + 1;
    elseif(gesType == 7)%2)
        fprintf(fid,'G.LEFT \t');
        totalLeft = totalLeft + 1;
    elseif(gesType == 3)
        fprintf(fid,'G.CIRC \t');
        totalCircle = totalCircle + 1;
    elseif(gesType == 4)
        fprintf(fid,'G.BOXX \t');
        totalBox = totalBox + 1;
    elseif(gesType == 5)
        fprintf(fid,'G.SMAL \t');
        totalSmall = totalSmall + 1;
    elseif(gesType == 6)
        fprintf(fid,'G.BIGG \t');
        totalBig = totalBig + 1;
    end
    
    
    %% Session 1
    posDiff = 0;
    otherPosDiff = 0 ;
    for idxStep = 130
        posDiff = posDiff + (position(idxStep,2) - position(1,2));
        %otherPosDiff = otherPosDiff + (position(idxStep,5) - position(1,5));
    end
    posDiff = posDiff / 1;%0;
    otherPosDiff = otherPosDiff / 1;%0;
    
    if(posDiff > 0.05)
        fprintf(fid,'SUCCESS\n');
        if(gesType == 8)%1)
            sRight = sRight + 1;
        elseif(gesType == 7)%2)
            sLeft = sLeft + 1;
        elseif(gesType == 3)
            sCircle = sCircle + 1;
        elseif(gesType == 4)
            sBox = sBox + 1;
        elseif(gesType == 5)
            sSmall = sSmall + 1;
        elseif(gesType == 6)
            sBig = sBig + 1;
        end
    else
        fprintf(fid,'FAILURE\t');
    end
    
    if(posDiff < 0.05)
        numF = numF +1;
        if(otherPosDiff >= 0.05)
            numConfusion = numConfusion + 1;
            fprintf(fid,'CONFUSIN\n');
        else
            fprintf(fid,'GRASPING\n');
        end
    else
        numS = numS +1;
        %         if(objX == 1 || objX == 2 || objX == 5)
        %             if(objType == 1)
        %                 numS_left_tall = numS_left_tall + 1;
        %             elseif(objType == 2)
        %                 numS_left_long = numS_left_long + 1;
        %             end
        %         else
        %             if(objType == 1)
        %                 numS_right_tall = numS_right_tall + 1;
        %             elseif(objType == 2)
        %                 numS_right_long = numS_right_long + 1;
        %             end
        %         end
        %
        %         if(gesType == 8)%1)
        %             numS_pnt_right = numS_pnt_right + 1;
        %         elseif(gesType == 7)%2)
        %             numS_pnt_left = numS_pnt_left + 1;
        %         elseif(gesType == 5)
        %             numS_obj_long = numS_obj_long + 1;
        %         elseif(gesType == 6)
        %             numS_obj_tall = numS_obj_tall + 1;
        %         end
        
        
    end
    
    
    
    
    
end
s1SR = (numS / (numS + numF) ) *100;
s2SR = (numS2 / (numS2 + numF2) ) *100;
sAllSR = (numSall / (numSall + numFall) ) *100;

fprintf(fid,'Success\tConfusion\tPNT\tOBJ\tLEFT\tRIGHT\tTALL\tLONG\n');
text = sprintf('%d/%d\t',numS,numS+numF);
fprintf(fid,text);


fprintf(fid,'%d\t%d\t%d\t%d\t%d\t%d\t%d\n', ...,
    numConfusion,numS_pnt_right+numS_pnt_left,numS_obj_long+numS_obj_tall, ...,
    numS_left_tall+numS_left_long,numS_right_tall+numS_right_long, ...,
    numS_left_tall+numS_right_tall, numS_left_long+numS_right_long);

text = sprintf('SESSION 1: %.2f [%d / %d]\n',s1SR,numS,numS+numF)
fprintf(fid,text);

fprintf(fid,'\n');
fprintf(fid,'Left\tRight\tBig\tSmall\tCircle\tBox\n');

text = sprintf('%d/%d\t',sLeft,totalLeft);
fprintf(fid,text);

text = sprintf('%d/%d\t',sRight,totalRight);
fprintf(fid,text);

text = sprintf('%d/%d\t',sBig,totalBig);
fprintf(fid,text);

text = sprintf('%d/%d\t',sSmall,totalSmall);
fprintf(fid,text);

text = sprintf('%d/%d\t',sCircle,totalCircle);
fprintf(fid,text);

text = sprintf('%d/%d\t',sBox,totalBox);
fprintf(fid,text);

fprintf(fid,'\n');

text = sprintf('%d\t%.2f\t',numS,s1SR);
fprintf(fid,text);

text = sprintf('%.2f\t',sLeft*100/totalLeft);
fprintf(fid,text);

text = sprintf('%.2f\t',sRight*100/totalRight);
fprintf(fid,text);

text = sprintf('%.2f\t',sBig*100/totalBig);
fprintf(fid,text);

text = sprintf('%.2f\t',sSmall*100/totalSmall);
fprintf(fid,text);

text = sprintf('%.2f\t',sCircle*100/totalCircle);
fprintf(fid,text);

text = sprintf('%.2f',sBox*100/totalBox);
fprintf(fid,text);

fclose(fid);
