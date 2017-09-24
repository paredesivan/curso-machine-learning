clear ; close all; clc

X=[ 
    1,1,1;
    1,2,sqrt(2);
    1,3,sqrt(3);
    1,4,sqrt(4);
    1,6,sqrt(6);
    1,8,sqrt(8);
    1,10,sqrt(10);
    1,15,sqrt(15)];

Y=[50;100;133;150;200;250;300;400];
theta=[-19;12;63];
alpha=0.03;
ite=10000;

%J = computeCost(X,Y,theta);
[theta,historial]=gradientDescent(X, Y, theta, alpha, ite);

plotData(X(:,2), Y);
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-'); %x * theta es la hipotesis
legend('Training data', 'Linear regression');
hold off;

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCost) and gradient here.
    %
    
    h = X * theta;
    % X' * (h - y) = sum((h - y) .* X)'
    theta=theta-alpha * (1 / m) * (X' * (h - y));
    
    %       dado A=[1,2;3,4];
    %            B=[]5,6;7,8;
    %            C=A.*B es la mulitplicacion elemento por elemento
    %            C(i,j)=a(i,j) * b(i,j);
    
    
    % ============================================================
    
    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
    
    
end

end

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

hipotesis = X*theta;
residuos = hipotesis - y;
cuadrados = residuos.^2;
sumatoria = sum(cuadrados);

J = sumatoria / (2*m);

% =========================================================================

end

