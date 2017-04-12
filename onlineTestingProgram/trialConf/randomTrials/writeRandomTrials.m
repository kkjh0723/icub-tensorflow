clear all; close all; clc;
numberOfRandomTesting = 50;

% ID for random obj (ABCD)
% A (OBJ): 3: ball, 4: box, 5: cylinder
% B (SIZE): 5: very small, 4: small, 3: middle, 2: big, 1: very big
% C (LOC): 1~4: far left to far right
% D (RandomTrial) 0~9

if(0)
    for A = 4:6
        for B = 1:5
            for C = 1:4
                for D = 0:9
                    objID = A*1000 + B*100 + C*10 + D;
                    fileName = sprintf('./randomTrial_%04d.txt',objID);
                    fid=fopen(fileName,'w');
                    
                    % Choose the random position
                    if(C == 1)
                        a = 0.11; b = 0.14;
                    elseif(C == 2)
                        a = 0.08; b = 0.11;
                    elseif(C == 3)
                        a = -0.01; b = 0.02;
                    elseif(C == 4)
                        a = -0.04; b = -0.01;
                    end
                    x = (b-a).*rand(1,1) + a;
                    fprintf(fid,'%.3f\t',x);
                    
                    if(D < 5)
                        a = 0.27; b = 0.30;
                    else
                        a = 0.24; b = 0.27;
                    end
                    y = (b-a).*rand(1,1) + a;
                    fprintf(fid,'%.3f\t',y);
                    
                    
                    
                    if(A == 4)
                        if(B == 5)
                            r = 0.025 * 0.75;
                        elseif(B == 4)
                            r = 0.025 * 1; % TRAIN DATA
                        elseif(B == 3)
                            r = 0.025 * 1.25;
                        elseif(B == 2)
                            r = 0.025 * 1.5; % TRAIN DATA
                        elseif(B == 1)
                            r = 0.025 * 1.75;
                        end
                        fprintf(fid,'%.3f\t',r);
                        
                        
                    elseif(A == 5)
                        if(B == 5)
                            x = 0.028 * 0.8;
                            y = 0.05 * 0.8;
                            z = 0.06 * 0.8;
                        elseif(B == 4)
                            x = 0.028 * 1; % TRAIN DATA
                            y = 0.05 * 1; % TRAIN DATA
                            z = 0.06 * 1; % TRAIN DATA
                        elseif(B == 3)
                            x = 0.028 * 1.2;
                            y = 0.05 * 1.2;
                            z = 0.06 * 1.2;
                        elseif(B == 2)
                            x = 0.028 * 1.4; % TRAIN DATA
                            y = 0.05 * 1.4; % TRAIN DATA
                            z = 0.06 * 1.4; % TRAIN DATA
                        elseif(B == 1)
                            x = 0.028 * 1.6;
                            y = 0.05 * 1.6;
                            z = 0.06 * 1.6;
                        end
                        fprintf(fid,'%.3f\t',x);
                        fprintf(fid,'%.3f\t',y);
                        fprintf(fid,'%.3f\t',z);
                        
                        
                        
                    elseif(A == 6)
                        if(B == 5)
                            r = 0.025 * 0.75;
                            l = 0.07 * 0.85;
                        elseif(B == 4)
                            r = 0.025 * 1; % TRAIN DATA
                            l = 0.07 * 1; % TRAIN DATA
                        elseif(B == 3)
                            r = 0.025 * 1.25;
                            l = 0.07 * 1.15;
                        elseif(B == 2)
                            r = 0.025 * 1.5; % TRAIN DATA
                            l = 0.07 * 1.3; % TRAIN DATA
                        elseif(B == 1)
                            r = 0.025 * 1.75;
                            l = 0.07 * 1.45;
                        end
                        fprintf(fid,'%.3f\t',r);
                        fprintf(fid,'%.3f\t',l);
                        
                    end
                    
                    
                    
                    fclose(fid);
                    
                end
            end
        end
    end
end

for A = 4:6
    subplot(3,1,A-3);
    for B = 2:2:4
        for C = 1:4
            for D = 0:9
                objID = A*1000 + B*100 + C*10 + D;
                fileName = sprintf('./randomTrial_%04d.txt',objID);
                rfile = load(fileName);
                x = rfile(1,1);
                y = rfile(1,2);
                
                scatter(x,y,'MarkerEdgeColor','b', 'MarkerFaceColor','b', 'LineWidth',B); hold on;
                %                 drawnow
            end
        end
    end
end
% scatter(0.07,0.3,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
% scatter(0.01,0.3,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
% scatter(-0.05,0.3,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
% scatter(-0.11,0.3,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
% scatter(0.07,0.24,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
% scatter(0.01,0.24,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
% scatter(-0.05,0.24,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
% scatter(-0.11,0.24,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
%
%
% scatter(0.04,0.27,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
% scatter(-0.02,0.27,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
% scatter(-0.08,0.27,'MarkerEdgeColor',[1 0 0], 'MarkerFaceColor',[1 0 0], 'LineWidth',5.5); hold on;
%
%
% xlim([-0.111 0.071]);
% ylim([0.239 0.301]);
% grid on;

