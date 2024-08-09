% SIS Model
function [t, y] = sis_model(i0, lambda, mu, tmax)
    [t, y] = ode45(@(t, y) [lambda * y(1) * (1 - y(1)) - mu * y(1)], [0 tmax], i0);
end