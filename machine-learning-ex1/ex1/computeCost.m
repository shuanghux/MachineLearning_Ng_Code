function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%if size(X) == [m 1]
%        X = [ones(m),X]; %add a column 1 ahead of input X matrix
%end
%Hx = zeros(m,1);

%Hx = X*theta;
%for i = 1:m
%    Hx(i) = theta(1)+theta(2)*X(i);
%    J = J + (Hx(i)-y(i))^2;
%end
%J = J/2/m;
J = 0;
for i = 1:m
    J = J + (theta' *  X(i,:)' - y(i))^2;
end

J = J/(2*m);

% =========================================================================

end
