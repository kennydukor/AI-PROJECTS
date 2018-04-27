function data_plot(X, y)
%now, we visualize the data by plotting. Since it is a classification
%problem, we have to distinguish the zeros and ones in vector y (let 
%1 be positive and 0 be negative)
figure; hold on;

pos = find(y == 1);
neg = find(y == 0);

%plotting data in the X matrix but differentiated by the positive and
%negative results
plot(X(pos, 1), X(pos, 2), 'k+', 'Markersize', 7, 'Linewidth', 2)
%'X(pos, 1)' means values of X on first column corresponding to positive 
%results. recall, X is an M x 2 matrix

hold on; %keeps the first plot on hold and adds second plot to it
fprintf('This is for the positive result area \n')
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
plot(X(neg, 1), X(neg, 2), 'ko', 'Markersize', 7, 'Linewidth', 2,...
    'MarkerFaceColor', 'y')