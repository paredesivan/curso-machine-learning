function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

 g = 1.0 ./ (1.0 + exp(-z));
 
 
%  mia
% g = 1 ./(exp((-z))+1);
end
