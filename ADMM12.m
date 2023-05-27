function [x, output] = ADMM12(A, b, pm)

%%    u = min_{x} ||x||_1 - ||x||_2
%%    s.t. ||Ax - b||_2 <= sigma
[M,N]       = size(A);
start_time  = tic;

if nargin<3
    x0 = zeros(N,1);
    alpha = 1;
    lambda = 10;
    tau = lambda;
    sigma = 1e-4;
    maxit = 1e4;
    reltol  = 1e-6;
else

    %% parameters
    if isfield(pm,'x0')
        if size(A, 2) == size(pm.x0,1)
            x0 = pm.x0;
        else
            x0 = zeros(N,1);
        end
    else
        x0 = zeros(N,1);
    end
    if isfield(pm,'alpha')
        alpha = pm.alpha;
    else
        alpha = 1;
    end
    if isfield(pm,'lambda')
        lambda = pm.lambda;
    else
        lambda = 10;  % default value
    end
    if isfield(pm,'tau')
        tau = pm.tau;
    else
        tau = lambda;
    end
    if isfield(pm,'sigma')
        sigma = pm.sigma;
    else
        sigma = 1e-4;
    end
    % maximum number of iterations
    if isfield(pm,'maxit')
        maxit = pm.maxit;
    else
        maxit = 1e4; % default value
    end
    if isfield(pm,'reltol')
        reltol = pm.reltol;
    else
        reltol  = 1e-6;
    end
end

% The proximal operator of ell1 - alpha * ell2
% Reference: Lou Y, Yan M (2018) Fast L1 − L2 minimization via a proximal operator.
%            Journal of Scientific Computing 74(2):767–785
    function x = prox(y,lambda,alpha)
        % min_x .5||x-y||^2 + lambda( ||x||_1- alpha ||x||_2 )
        x = zeros(size(y));

        if max(abs(y)) > 0
            if max(abs(y)) > lambda
                x   = max(abs(y) - lambda, 0).*sign(y);
                x   = x * (norm(x,2) + alpha * lambda)/norm(x,2);
            else
                if max(abs(y))>=(1-alpha)*lambda
                    [~, i]  = max(abs(y));
                    x(i(1)) = (y(i(1)) + (alpha - 1) * lambda) * sign(y(i(1)));
                end
            end
        end

    end

proj = @(u) u .* min(sigma/norm(u,2), 1);
objective = @(u) norm(u, 1) - norm(u, 2);
constraint = @(u) norm(A*u - b, 2);
output.pm = pm;

%% pre-computing/initialize
xold    = x0;
y       = zeros(M,1);
z       = zeros(N,1);
p1      = zeros(M,1);
p2      = zeros(N,1);

% AtA     = A'*A;
% L       = chol(speye(N)*tau + lambda*AtA, 'lower');
% L       = sparse(L);
% U       = sparse(L');
I_plus_lamAAT = tau * speye(M) + lambda * (A*A');
AtA_plus_I_inv = (1/tau)*(speye(N) - lambda * (A'*(I_plus_lamAAT\A)));

for it =1: maxit
    %update x
    rhs = lambda*(A'*(y+b)) + A'*p1 + tau*z + p2;
    %     x   = U\(L\rhs);
    x = AtA_plus_I_inv * rhs;

    %update y
    y = proj(A*x - b - p1/lambda);

    %update z
    z = prox(x - p2/tau, 1/tau, alpha);

    %update p
    p1 = p1 - lambda*(A*x - y - b);
    p2 = p2 - tau*(x-z);

    % stop conditions & outputs
    relerr      = norm(xold - x, 2)/max([norm(xold, 2), norm(x, 2), eps]);

    output.relerr(it)    = relerr;
    output.objective(it) = objective(x);
    output.time(it)      = toc(start_time);
    output.constraint(it)= constraint(x);
    output.xoutput(:,it) = x;

    if relerr < reltol && it > 2
        break;
    end
    xold = x;
end
end

