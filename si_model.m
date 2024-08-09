% SI Model
function [t, y] = si_model(i0, lambda, tmax)
    [t, y] = ode45(@(t, y) [lambda * y(1) * (1 - y(1))], [0 tmax], i0);
end
