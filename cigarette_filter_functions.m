function varargout = cigarette_filter_functions(varargin)
    % Wrapper function to access various subfunctions.
    % This function returns output based on the number of input arguments.
    % Input:
    %   varargin - Variable input arguments.
    % Output:
    %   varargout - Variable output based on the subfunction called.
    
    if nargin == 7
        % Calculate total toxin intake Q
        varargout{1} = calculate_Q(varargin{:});
    elseif nargin == 4
        % Calculate Q ratio
        varargout{1} = calculate_Q_ratio(varargin{:});
    elseif nargin == 3
        % Plot Q vs a specified parameter
        plot_Q_vs_parameter(varargin{:});
    else
        error('Invalid number of input arguments. Please check your inputs.');
    end
end

function Q = calculate_Q(a, M, beta, l2, v, b, l1)
    % Calculate the total toxin intake Q
    % Input:
    %   a, M, beta, l2, v, b, l1 - Parameters as defined in the problem
    % Output:
    %   Q - Total toxin intake
    
    if l1 <= 0 || l2 <= 0 || v <= 0
        error('Parameters l1, l2, and v must be positive.');
    end
    
    r = (1 - a) * b * l1 / v;
    Q = a * M * exp(-beta * l2 / v) * (1 - exp(-r)) / r;
end

function Q_ratio = calculate_Q_ratio(beta, b, l2, v)
    % Calculate the ratio of toxin intake with and without the filter
    % Input:
    %   beta, b, l2, v - Parameters as defined in the problem
    % Output:
    %   Q_ratio - The ratio of toxin intake
    
    Q_ratio = exp(-(beta - b) * l2 / v);
end

function plot_Q_vs_parameter(param_name, param_range, fixed_params)
    % Plot Q vs a specified parameter
    % Input:
    %   param_name - Name of the parameter to vary ('l2' or 'beta')
    %   param_range - Range of values for the parameter
    %   fixed_params - Structure containing fixed parameter values
    
    Q_values = zeros(size(param_range));
    
    for i = 1:length(param_range)
        if strcmp(param_name, 'l2')
            Q_values(i) = calculate_Q(fixed_params.a, fixed_params.M, fixed_params.beta, ...
                                      param_range(i), fixed_params.v, fixed_params.b, fixed_params.l1);
        elseif strcmp(param_name, 'beta')
            Q_values(i) = calculate_Q(fixed_params.a, fixed_params.M, param_range(i), ...
                                      fixed_params.l2, fixed_params.v, fixed_params.b, fixed_params.l1);
        else
            error('Invalid parameter name. Use ''l2'' or ''beta''.');
        end
    end
    
    figure;
    plot(param_range, Q_values, 'LineWidth', 1.5);
    xlabel(param_name, 'Interpreter', 'none');
    ylabel('Q (Total toxin intake)');
    title(['Q vs ', param_name], 'Interpreter', 'none');
    grid on;
end
