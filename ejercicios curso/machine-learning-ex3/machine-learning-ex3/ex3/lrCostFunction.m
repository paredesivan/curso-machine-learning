function [J, grad] = lrCostFunction(theta, X, y, lambda)

%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

hipotesis = sigmoid(X * theta);

diferencias = y .* log(hipotesis) + ((1-y) .* log( 1 - hipotesis));

acumulado = sum(diferencias); 

% costo sin regularizar
J = acumulado / (-m);
% fprintf('J es %f',J); 

grad = (1 / m) * (X' * (hipotesis - y));
% es lo mismo grad = 1/m .* (sum((hipotesis - y) .* X)');

% quito theta(0) porque theta(0) no se debe regularizar
thetaSin0= [0 ; theta(2:size(theta), :)];

% elevo cada theta al cuadrado
suma = sum(thetaSin0.^2);

% funcion costo regularizada
J = J + ((lambda*suma) / (2*m));

% gradiente regularizado
grad = grad + (lambda/m).*thetaSin0;

end
