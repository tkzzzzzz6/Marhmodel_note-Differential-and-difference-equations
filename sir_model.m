% SIR Model
function [t, y] = sir_model(i0, s0, lambda, mu, tmax)
    [t, y] = ode45(@(t, y) [
        lambda * y(1) * y(2) - mu * y(1)  % di/dt
        -lambda * y(1) * y(2)  % ds/dt
        mu * y(1)  % dr/dt
    ], [0 tmax], [i0; s0; 1-i0-s0]);
end