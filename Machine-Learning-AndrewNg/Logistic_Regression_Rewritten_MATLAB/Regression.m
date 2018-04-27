%Dukor Kenechi Franklin, kennydukor@gmail.com
%==========================================================================
%first we load the data from the txt file. It contains two
%features; say X1 and X2 and an out put y. Let the feature be an M x 2
%matrix. y is a classification output containing 0 and 1

data = load('ex2data1.txt');

%we then assign the loaded data to X and y

X = data(: , [1,2]); %'(:)'all rows in '(,[1,2])' column 1 and 2 
y = data(: , 3); %'(:)'all rows in '(,[3])' column 3

%===============================Visualization==============================
data_plot(X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off; %disables the hold on

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%=======================Visualization Complete=============================

%Please review Andrew Ng's course on machine learning for the intuition and
%formulars for logistic regression

% Lets shift our cost function and gradient formula calculation to another 
%file
%-------------------------- Cost Function ---------------------------------

%We have to add an additional x feature (X0) of ones in the first column. 
%this is because of the theta0 parameter present in the hypothesis equation.

[m,n] = size(X); 

X = [ones(m, 1) X]; %"[(ones(m,1))"- create ones of row length m, " X]" and 
                    % embed to X

%------------
%Lets define the initial parameter(theta) and set them to zero. it should
%an n+1 column. ie 3 column vector for our study
theta_initial = zeros(n + 1, 1);

[cost, grad] = Cost_Gradient_formula(theta_initial, X, y);
fprintf('The cost at initial theta of zeros is: %f\n', cost);
fprintf('\nProgram paused. Press enter to continue to see gradient.\n\n');
pause;
fprintf('===================================================\n');
fprintf('The gradient at initial theta of zeros is: %f\n\n', grad);
fprintf('\nProgram paused. Press enter to continue to test at theta [-24; 0.2; 0.2].\n');
fprintf('===================================================\n');
pause;

%Now, lets test for a different set of theta
theta_test = [-24; 0.2; 0.2];
[cost, grad] = Cost_Gradient_formula(theta_test, X, y);
fprintf('for theta values of -24, 0.2 and 0.2 as theta 1,2,3 cost is: %f\n', cost);
fprintf('\nProgram paused. Press enter to continue to see gradient.\n\n');
pause;
fprintf('===================================================\n');
fprintf('for theta values of -24, 0.2 and 0.2 as theta 1,2,3 gradient is: %f\n\n', grad);
fprintf('\nProgram paused. Press enter to continue to use fminunc to minimize.\n');
fprintf('===================================================\n');
pause;

%Using the fminunc (minimum of unconstrained function) to minimize our cost
options = optimset('GradObj', 'on', 'MaxIter', 400); %Optimization settings

%Defining the function to optimize
[theta, cost] = ...
	fminunc(@(t)(Cost_Gradient_formula(t, X, y)), theta_initial, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('===================================================\n');
fprintf('\nProgram paused. Press enter to continue to see theta values.\n');
pause;
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('===================================================\n');

% Plot Data
data_plot(X(:,2:3), y);
hold on;

%Yet to be explained==================================

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off;




