%-------------------- Cost Function and Gradient Formula-------------------
function [J, grad] = Cost_Gradient_formula(theta, X, y)

%Lets define 'm'

m = length(y); % m is the total number of training examples

% defining sigmoid function for the logistic regression (g)
h = 1 ./ (1 + exp(-1 .* (X * theta)));
%the cost is then given by 'J'
J = 1/m .* sum((-y)' * log(h) - (1-y)' * log(1 - h));
%the gradient is also given  by 'grad'
grad = 1/m * (X' * (h - y));
