% I Model
function [t, i] = i_model(i0, lambda, tmax)
    [t, i] = ode45(@(t, i) lambda * i, [0 tmax], i0);
end