function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% llamo a la funcion de costo
[J, grad] = costFunction(theta, X, y);

% quito theta(0) porque theta(0) no se debe regularizar
thetaSin0= [0 ; theta(2:size(theta), :)];

% elevo cada theta al cuadrado
suma = sum(thetaSin0.^2);

% funcion costo regularizada
J = J + ((lambda*suma) / (2*m));

% gradiente regularizado
grad = grad + (lambda/m).*thetaSin0;


% =============================================================

end
