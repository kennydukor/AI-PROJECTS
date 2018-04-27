%Linear regression with one feature
%First, we have to import the data from data1
data = load('data1.txt');
%assigning the information to X and y
X = data(:,1); y = data(:,2);

m = length(y); %setting m (no of training example) to be the number of examples in y
%============================Plotting the Data=============================
plot_data(X, y);
fprintf('Program paused. Press enter to calculate cost (J).\n');
pause;
%==========================================================================
[m,n] = size(X);

X = [ones(m , 1) X]; %"[(ones(m,1))"- create ones of row length m, " X]" and 
                    % embed to X
%--------------------------------------------------------------------------
%Lets define the initial parameter(theta) and set them to zero. it should
%an n+1 column. ie 3 column vector for our study
theta_initial = zeros(n+1, 1); 

%------------------------------------------
%Calculating for the cost
iterations = 1500;
alpha = 0.01;

[J, grad, all_J] = cost_gradient(theta, X, y, m, alpha, iterations); %J is cost function
fprintf('The cost at initial theta of zeros is: %f\n', J);
fprintf('\nProgram paused. Press enter to see gradient.\n\n');
pause;

fprintf('The theta gotten with gradient descent is: %f\n', grad);
fprintf('\nProgram paused. Press enter to see values of J at each iteration.\n\n');
pause;

fprintf('The cost at each iteration are: %f\n', all_J);
fprintf('\nProgram paused. Press enter to continue.\n\n');
pause;
 


