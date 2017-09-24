% fprintf('Loading data ...\n');
% data = load('ex1data2.txt');
% X = data(:, 1:2);
% y = data(:, 3);
% m = length(y);
% % 
% [X_norm mu sigma] = featureNormalize(X);

function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: 
% para cada variable:
%     calcular su promedio
%     restale el dato
%     guardar el promedio en mu
%     calcular su desvio
%     dividir cada dato por su desvio
%     guardar el desvio en sigma
    
%First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%  
% G=[5;6;7];
% fprintf('%i\n', G);
% V = G-3;
% fprintf('v es\n');
% fprintf('%i\n', V);
% x1 = X(:, 1);
% fprintf('x1 es %f\n',x1);
% x2 = X(:, 2);

[m,n]=size(X);
for i=1:n
    mu(1,i)= mean(X(:,i)); %promedio de la columna 1 
%     fprintf('mu o valor promedio de la columna %i, es: %f\n',i,mu(1,i));
%     fprintf('la columna %f\n',X(:, i)-mu(1,i));
    X_norm(:,i)=X(:, i)-mu(1,i);
    sigma(1,i) = std(X(:,i)); %desvio standar o max-min de la columna 1
%     fprintf('sigma %f\n',sigma);
    X_norm(:,i)=X_norm(:, i)/sigma(1,i);
end

% ============================================================

end




