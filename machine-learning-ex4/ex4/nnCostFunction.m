function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
%% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%Feedforward
yd = eye(num_labels);
y = yd(y,:); %5000*10
a1 = [ones(size(X,1),1) X]; %5000*401
z2 = a1 * Theta1'; %5000*25
a2 = [ones(size(z2,1),1) sigmoid(z2)]; %5000*26
z3 = a2 * Theta2'; %5000*10
a3 = sigmoid(z3);

J_pre = y.*log(a3)+((1-y).*log(1-a3));
J = -sum(sum(J_pre))/m;

%Regula
theta1_reg = Theta1(:,2:end);
theta2_reg = Theta2(:,2:end);

reg = lambda/2/m*(sum(sum(theta1_reg.^2))+sum(sum(theta2_reg.^2)));
J = J + reg;

%% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for t = 1:m
   a1 = [1 X(t,:)]'; %401*1
   z2 = Theta1*a1; %25*1
   a2 = [1; sigmoid(z2)]; %26*1
   z3 = Theta2*a2; %10*1
   a3 = sigmoid(z3); %10*1
   Y = y(t,:); %1*10
   delta3 = a3 - Y';%10*1
   delta2 = (Theta2' * delta3).*[1;sigmoidGradient(z2)];%([26*10]*[10*1]).*(26*1)
   delta2 = delta2(2:end,:);%25*1
   D1 = D1 + delta2*a1'; %(25*401)+(25*1 * 1*401)
   D2 = D2 + delta3*a2';
end
Theta1_grad = (1/m) * D1 + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * D2 + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
%% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
