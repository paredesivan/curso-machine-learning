theta=[1;4;6];
lambda=2;
m=length(theta);

theta=theta.^2;
suma=sum(theta);
total=(lambda*suma)/(2*m);

fprintf('theta nuevo es:%d y la suma es:%d, y multiplicado por lambda/2m es:%d\n',theta,suma,total);

subtheta=theta(1:length(theta));
fprintf('solo desde el indice2 %d',subtheta);