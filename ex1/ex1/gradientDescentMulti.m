function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %



    sizeTheta = size(theta);
    tmpTheta = zeros(sizeTheta(1,1), sizeTheta(1,2));
    for thetaI = 1 : size(theta)
        derivaJ = 0;
        for i = 1 : m
            hx = 0;
            for j = 1 : size(theta)
                hx = hx + X(i, j) * theta(j);
            end
            derivaJ = derivaJ + (hx - y(i)) * X(i, thetaI);
        end
        tmpTheta(thetaI, 1) = theta(thetaI, 1) - alpha / m * derivaJ;
    end
    theta = tmpTheta;








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
