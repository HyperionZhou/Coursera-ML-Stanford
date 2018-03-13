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
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
X = [ones(m, 1) X];
hTheta = Theta1 * X';
hTheta = sigmoid(hTheta);
hTheta = [ones(1, size(hTheta, 2)); hTheta];
hTheta = Theta2 * hTheta;
hTheta = sigmoid(hTheta);


newY = zeros(num_labels, m);
for i = 1 : m
    newY(y(i), i) = 1;
end

J = 1 / m * sum(sum(0 - newY .* log(hTheta) - (1 - newY) .* log(1 - hTheta)));




% Part 2: Implement the backpropagation algorithm to compute the gradients
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

maxSize = max([input_layer_size hidden_layer_size num_labels]);
layerSize = [input_layer_size, hidden_layer_size, num_labels];
L = 3;
X = X(:, 2 : end);
acDleta = zeros(3, maxSize + 1, maxSize + 1);
for t = 1 : m
    a = zeros(maxSize, L);
    a(1 : layerSize(1), 1) = X(t, :)';
    %feed forward propagation
    tmpA = [1; a(:, 1)];
    z = Theta1 * tmpA(1 : 1 + layerSize(1));
    a(1 : layerSize(2), 2) = sigmoid(z);
    tmpB = [1; a(:, 2)];
    z = Theta2 * tmpB(1 : 1 + layerSize(2));
    a(1 : layerSize(3), 3) = sigmoid(z);
    
    %feed back propagation
    delta = zeros(maxSize + 1, L);
    delta(1 : layerSize(3), 3) = a(1 : layerSize(3), 3) - newY(:, t);
    tmpA = [1; a(:, 1)];
    tmpB = [1; a(:, 2)];
    delta(1 : layerSize(2) + 1, 2) = (Theta2' * delta(1 : layerSize(3), 3))...
        .* tmpB(1 : 1 + layerSize(2))...
        .* (1 - tmpB(1 : 1 + layerSize(2)));
    delta(1 : layerSize(1) + 1, 1) = (Theta1' * delta(1 : layerSize(2), 2))...
        .* tmpA(1 : 1 + layerSize(1))...
        .* (1 - tmpA(1 : 1 + layerSize(1)));
    
    a = [ones(1, size(a, 2)); a];
    a(1 : layerSize(3), 3) = a(2 : layerSize(3) + 1, 3);
    %accumulate error
    
    
    
    for l = 1 : 2
        for i = 1 : layerSize(l + 1)
            for j = 1 : layerSize(l) + 1
                if l == 1
                    Theta1_grad(i, j) = Theta1_grad(i, j) + a(j, l) * delta(1 + i, l + 1);
                end
                if l == 2
                    Theta2_grad(i, j) = Theta2_grad(i, j) + a(j, l) * delta(i, l + 1);
                end
            end
        end
    end
    
    
end


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

tmpTheta1 = Theta1;
tmpTheta1(:, 1) = 0;
tmpTheta2 = Theta2;
tmpTheta2(:, 1) = 0;

J = J + lambda * (sum(sum(tmpTheta1 .^ 2)) + sum(sum(tmpTheta2 .^ 2))) / (2 * m);


Theta1_grad = Theta1_grad / m + lambda / m * [zeros(size(Theta1, 1), 1) Theta1(:, 2 : end)];
Theta2_grad = Theta2_grad / m + lambda / m * [zeros(size(Theta2, 1), 1) Theta2(:, 2 : end)];














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
