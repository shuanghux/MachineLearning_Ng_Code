function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% theta 2*1
% X m*2
% add a column at the beginning of X matrix to match theta0

hx = X*theta;
J = 1/(2*m)*sum((hx-y).^2)+lambda/(2*m)*sum(theta(2:end,1).^2);


grad = (1/m)*(X'*(hx-y)); % 2*12 * 12*1 = 2*1
grad = grad + lambda/m*[0;theta(2:end,:)];
grad = grad(:);







% =========================================================================

grad = grad(:);

end
