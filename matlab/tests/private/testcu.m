function [mrec, mmin, output] = testcu(solvers, options)

solvers = lower(solvers);

% Default options
rhobeg = 1;
rhoend = 1e-6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Suppose that MAXN is the maximal possible dimension of problems
% in our test. Ideally, we should set maxfun to maxfun_dim*MAXN.
% Note that MAXN is not maxdim, which may be overridden by options.
% If MAXN = 200, then we should set maxfun to 200*maxfun_dim.
% To be done later.
%maxfun_dim = 100;
%maxfun = 20000;
maxfun_dim = 200;
maxfun = 40000;
%maxfun_dim = 500;
%maxfun = 100000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% maxit is not used by Powell's methods
maxit = 1000;

ftarget = -inf;
perm = false;
randomizex0 = 0;
eval_options = struct();
nr = 5;
ctol = 1e-10;
cpenalty = 1e10;
%ctol = 1e-8;
%cpenalty = 1e8;
type = 'ubln'; % The default types of problems to test
mindim = 1; % The default minimal dimension of problems to test
if any(startsWith(solvers, 'cobyla'))
    maxdim = 20; % The default maximal dimension of problems to test
else
    maxdim = 50; % The default maximal dimension of problems to test
end
mincon = 0; % The default minimal number of constraints of problems to test
maxcon = 20000; % The default maximal number of constraints of problems to test, which is 100*MAXN
sequential = false;
% It is better to set debug and chkfunval to true. We do not want to profile solvers that have bugs.
% Note that the optimized version (compiled with -O) of prima will be called even if debug is true,
% because we compile only that version when profiling.
debug = true;
chkfunval = true;
output_xhist = true;
output_nlchist = true;
thorough_test = 0;
% minip is the minimal index of the problem to test. It is used if we want to skip the first few
% problems for debugging.
minip = 1;
maxip = 2^32 - 1;
strict = 2;

% Set options
options = setopt(options, rhobeg, rhoend, maxfun_dim, maxfun, maxit, ftarget, perm, randomizex0, ...
    eval_options, nr, ctol, cpenalty, type, mindim, maxdim, mincon, maxcon, ...
    sequential, debug, chkfunval, output_xhist, output_nlchist, thorough_test, minip, maxip, strict);

% Select the problems to test.
if isfield(options, 'list')
    plist = options.list; % Use the list provided by the user, neglecting all other requirements
    if (ischstr(plist))  % In case plist is indeed the name of a problem
        plist = {plist};
    end
else
    requirements = struct();
    requirements.mindim = options.mindim;
    requirements.maxdim = options.maxdim;
    requirements.mincon = options.mincon;
    requirements.maxcon = options.maxcon;
    requirements.type = options.type;

    if isfield(options, 'blacklist')
        requirements.blacklist = options.blacklist;
    else
        requirements.blacklist = {};
    end
    requirements.blacklist = [requirements.blacklist, black_list(solvers{1}), black_list(solvers{2})];

    plist = secup(requirements);
end

np = length(plist);
ns = length(solvers);
nr = options.nr;
maxfun = options.maxfun;
sequential = options.sequential;
minip = options.minip;
maxip = min(np, options.maxip);
strict = options.strict;

% These arrays will record the function values and constraint values during the tests.
pdim = NaN(np, 1);  % Data profile needs the dimension of the problem.
frec = NaN(np, ns, nr, maxfun + 1);
crec = NaN(np, ns, nr, maxfun + 1);
% Why the last dimension of frec and cref are maxfun + 1? Because the frec(ip, is, ir, nf+1) and
% cref(ip, is, ir, nf+1) will be used to record the function value and constraint violations of the
% x returned by the solver. Such x is important because users are supposed to take it as the
% solution. This is why we profile the solvers according to its function value and constraint
% violation when `natural_stop` is true. See perfdata.m and perfprof.m for details.

% These arrays will record the reference function values and constraint values when there is an
% eval_options or `randomizex0` is positive.
fref = NaN(np, ns, maxfun + 1);
cref = NaN(np, ns, maxfun + 1);

permuted = options.perm;
have_eval_options = ~isempty(fieldnames(options.eval_options));
eval_options = options.eval_options;
randomizex0 = abs(options.randomizex0);
ref_options = rmfield(options, {'perm', 'randomizex0', 'eval_options'});
use_ref = (have_eval_options || randomizex0 > 0);

% `eval_options` and `randomizex0` can occur at the same time, but neither of them are compatible
% with `perm`.
assert(~permuted || ~(have_eval_options || randomizex0));


fprintf('\n\nThe testing options:\n')
display(options);
fprintf('\n\nThe evaluation options:\n')
display(eval_options);
if isfield(eval_options, 'noise')
    display(eval_options.noise);
end
if isfield(eval_options, 'dnoise')
    display(eval_options.dnoise);
end

if sequential
    for ip = minip : maxip
        orig_warning_state = warnoff(solvers);

        pname = plist{ip};

        fprintf('\n%3d. \t%s:\n', ip, upper(pname));

        prob = macup(pname);
        orig_prob = prob;
        prob.orig_objective = prob.objective;
        prob.orig_nonlcon = prob.nonlcon;
        prob.orig_x0 = prob.x0;
        pdim(ip) = length(prob.x0);

        if use_ref
            %fprintf('\nCalculate fref and cref\n');
            for is = 1 : ns
                [fref(ip, is, :), cref(ip, is, :)] = testsolv(solvers{is}, prob, ref_options);
            end
        end

        rng(ip); permutations = get_perms(nr, length(prob.x0));

        for ir = 1 : nr
            if have_eval_options
                prob.objective = @(x) evalfun(prob.orig_objective, x, eval_options, ir);
                if ~isempty(prob.orig_nonlcon)
                    prob.nonlcon = @(x) evalcon(prob.orig_nonlcon, x, eval_options, ir);
                end
            end

            if randomizex0 > 0
                rng(ir); r = randn(length(prob.x0), 1);
                prob.x0 = prob.orig_x0 + randomizex0*norm(prob.orig_x0)*r/norm(r);
            end

            if permuted
                prob = permprob(orig_prob, permutations(ir, :));
                prob.orig_objective = prob.objective;
                prob.orig_nonlcon = prob.nonlcon;
            end

            for is = 1 : ns
                [frec(ip, is, ir, :), crec(ip, is, ir, :)] = testsolv(solvers{is}, prob, options);
            end
        end

        decup(prob);

        warning(orig_warning_state); % Restore the behavior of displaying warnings

    end
else
    parfor ip = minip : maxip
        orig_warning_state = warnoff(solvers);

        pname = plist{ip};

        fprintf('\n%3d. \t%s:\n', ip, upper(pname));

        prob = macup(pname);
        orig_prob = prob;
        prob.orig_objective = prob.objective;
        prob.orig_nonlcon = prob.nonlcon;
        prob.orig_x0 = prob.x0;
        pdim(ip) = length(prob.x0);

        if use_ref
            %fprintf('\nCalculate fref and cref\n');
            for is = 1 : ns
                [fref(ip, is, :), cref(ip, is, :)] = testsolv(solvers{is}, prob, ref_options);
            end
        end

        rng(ip); permutations = get_perms(nr, length(prob.x0));

        for ir = 1 : nr
            if have_eval_options
                prob.objective = @(x) evalfun(prob.orig_objective, x, eval_options, ir);
                if ~isempty(prob.orig_nonlcon)
                    prob.nonlcon = @(x) evalcon(prob.orig_nonlcon, x, eval_options, ir);
                end
            end

            if randomizex0 > 0
                rng(ir); r = randn(length(prob.x0), 1);
                prob.x0 = prob.orig_x0 + randomizex0*norm(prob.orig_x0)*r/norm(r);
            end

            if permuted
                prob = permprob(orig_prob, permutations(ir, :));
                prob.orig_objective = prob.objective;
                prob.orig_nonlcon = prob.nonlcon;
            end

            for is = 1 : ns
                [frec(ip, is, ir, :), crec(ip, is, ir, :)] = testsolv(solvers{is}, prob, options);
            end
        end

        decup(prob);

        warning(orig_warning_state); % Restore the behavior of displaying warnings

    end
end


% For uniformity of the code, we define fref, cref, and mref by the first random test if use_ref is false.
if ~use_ref
    assert(all(all(all(isnan(fref)))) && all(all(all(isnan(cref)))));
    for ip = 1 : np
        for is = 1 : ns
            fref(ip, is, :) = frec(ip, is, 1, :);
            cref(ip, is, :) = crec(ip, is, 1, :);
        end
    end
end

% Set cref(:, :, 1) to realmax if they are NaN.
for ip = 1:np
    cref(ip, isnan(cref(ip, :, 1)), 1) = realmax;
end

% Modify crec and cref.
% For the ip-th problem:
% 1. All values of crec/cref that are less than options.ctol*min(0.01, cref(ip, 1, 1)) are set to 0, meaning
% that we consider such constraint violations as zero. Other values are also reduced by this threshold.
% 2. All values of crec/cref that are more than max(0.1, 2*cref(ip, 1, 1)) are set to Inf, meaning that
% consider the corresponding iterates is too bad to consider.
for ip = 1 : np
    %cshift = options.ctol*min(0.01, cref(ip, 1, 1));
    %cshift = max(eps, options.ctol*min(0.01, cref(ip, 1, 1)));  % max(eps, ...) makes a difference.
    cshift = max(eps, options.ctol);  % ctol is considered as an absolute tolerance
    cref(ip, :, :) = max(0, cref(ip, :, :) - cshift);
    crec(ip, :, :, :) = max(0, crec(ip, :, :, :) - cshift);
    cbig = max(0.1, 2*cref(ip, 1, 1));
    for is = 1 : ns
        cref(ip, is, cref(ip, is, :) >= cbig) = Inf;
        for ir = 1 : nr
            crec(ip, is, ir, crec(ip, is, ir, :) >= cbig) = Inf;
        end
    end
end

% Define mrec
mrec = frec + options.cpenalty*crec;
% Prevent mrec(:,:,:,1) from being NaN or Inf.
for ip = 1 : np
    for is = 1 : ns
        for ir = 1 : nr
            if isnan(mrec(ip, is, ir, 1)) || mrec(ip, is, ir, 1) >= Inf
                mrec(ip, is, ir, 1) = realmax;
            end
        end
    end
end

% Define mref_min.
mref = fref + options.cpenalty*cref;  % The entries of mref is not used below. We need only mref_min.
mref_min = min(min(mref, [], 3), [], 2);  % Minimum of mref for each problem.


% Define mmin, which is used as the "minimum merit function value" for problem ip. However, we
% have different definitions of mmin depending on the strictness as follows.
% N.B.: When mref_min is not taken into account, it may happen that the solvers "solve more problems
% when there is noise / perturbation than when there is not". This is because the value of mmin may
% be much larger in the former case. This should be noted particularly if the functions are
% evaluated in single precision.
if strict == 0
    % Minimum taken over the current run.
    mmin = min(min(mrec, [], 4), [], 2);
elseif strict == 1
    % Minimum taken over the current run and the reference run.
    mrec_min = min(min(mrec, [], 4), [], 2);
    mmin = min(mrec_min, mref_min);  % Works, even though the two arrays have different sizes.
elseif strict == 2
    % Minimum taken over all the random runs.
    mmin = min(min(min(mrec, [], 4), [], 3), [], 2);
else
    % Minimum taken over all the random runs and the reference run.
    mrec_min = min(min(min(mrec, [], 4), [], 3), [], 2);
    mmin = min(mrec_min, mref_min);
end

output = struct();
output.plist = plist;
output.pdim = pdim;

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fval_history, cv_history, output] = testsolv(solver, prob, options)

prob.options = setsolvopt(solver, length(prob.x0), options); % Set the options for the solver

if ischstr(solver)
    prob.options.classical = endsWith(solver, '_classical');
    if endsWith(solver, '_single')
        prob.options.precision = 'single';
    end
    if endsWith(solver, '_quadruple')
        prob.options.precision = 'quadruple';
    end
    % `regexprep` removes '_classical' in case 'solver' ends with it. Similar for '_single', '_quadruple'.
    solver = regexprep(solver, '_classical$', '');
    solver = regexprep(solver, '_single$', '');
    solver = regexprep(solver, '_quadruple$', '');
    solver = regexprep(solver, '_base$', '_last');
    solver = str2func(solver);
end

% N.B.: prob.options.maxfun may differ from options.maxfun. Here we use the latter. Otherwise, the
% sizes will mismatch when we assign the result of this function to frec and crec.
maxfun = options.maxfun;
fval_history = NaN(1, maxfun + 1);
cv_history = NaN(1, maxfun + 1);

%have_eval_options = isfield(options, 'eval_options') && isstruct(options.eval_options) && ~isempty(fieldnames(options.eval_options));
prob.options.output_xhist = true;  % We always need xhist to recover the history of the computation.

[x, ~, ~, output] = solver(prob);
% Solvers (e.g., fmincon) may not respect maxfun. Indeed, PRIMA solvers may also increase maxfun
% if it is too small (e.g., <= npt for NEWUOA).
nf = min(maxfun, output.funcCount);

if (nf >= 1)
    % Use xhist and the original data of the problem to get fval_history and cv_history. Do NOT use
    % the information returned by the solver, as the solver may change the data (e.g., lincoa
    % may modify the right-hand side of linear constraints when x0 is infeasible; in addition, it
    % scales the constraints so that their gradients have norm 1), making results not comparable.
    xhist_cell = num2cell(output.xhist(:, 1:nf), 1);
    fval_history(1:nf) = cellfun(prob.orig_objective, xhist_cell);
    orig_cstrv = @(x) get_cstrv(x, prob.Aineq, prob.bineq, prob.Aeq, prob.beq, prob.lb, prob.ub, prob.orig_nonlcon);
    cv_history(1:nf) = cellfun(orig_cstrv, xhist_cell);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set the (nf+1)-th entries of fval_history and cv_history to the values returned by the solver.
    % This is needed for plotting when natrual_stop is true.
    fval_history(nf + 1) = prob.orig_objective(x);
    cv_history(nf + 1) = orig_cstrv(x);
    % Removing the following two lines, we keep the trailing entries of fval_history and cv_history
    % being NaN. This is correct and needed for plotting when natrual_stop is true.
    %fval_history(nf+2:maxfun) = fval_history(nf);
    %cv_history(nf+2:maxfun) = cv_history(nf);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    % Sometimes PRIMA may return nf = 0, e.g., when it detects infeasibility.
    fval_history = prob.f0;
    cv_history = prob.constrv0;
end

cv_history(isnan(cv_history)) = Inf;
assert(all(cv_history >= 0));

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function options = setopt(options, rhobeg, rhoend, maxfun_dim, maxfun, maxit, ftarget, perm, ...
        randomizex0, eval_options, nr, ctol, cpenalty, type, mindim, maxdim, mincon, maxcon, ...
        sequential, debug, chkfunval, output_xhist, output_nlchist, thorough_test, minip, maxip, strict) % Set options

if (~isfield(options, 'rhoend'))
    options.rhoend = rhoend;
end
if (~isfield(options, 'rhobeg'))
    options.rhobeg = rhobeg;
end
if (~isfield(options, 'maxit'))
    options.maxit = maxit;
end
if (~isfield(options, 'ftarget'))
    options.ftarget = ftarget;
end
if (~isfield(options, 'ctol'))
    options.ctol = ctol;
end
if (~isfield(options, 'cpenalty'))
    options.cpenalty = cpenalty;
end
if ~isfield(options, 'perm')
    options.perm = perm;
end
options.perm = logical(options.perm);
if (~isfield(options, 'randomizex0'))
    options.randomizex0 = randomizex0;
end
options.randomizex0 = abs(options.randomizex0);
if (~isfield(options, 'nr'))
    options.nr = nr;
end
if (~isfield(options, 'type'))
    options.type = type;
end
if (~isfield(options, 'mindim'))
    options.mindim = mindim;
end
if (~isfield(options, 'maxdim'))
    options.maxdim = maxdim;
end
if (~isfield(options, 'mincon'))
    options.mincon = mincon;
end
if (~isfield(options, 'maxcon'))
    options.maxcon = maxcon;
end
options.maxcon = min(options.maxcon, 100*options.maxdim);
if (~isfield(options, 'maxfun_dim'))
    options.maxfun_dim = maxfun_dim;
end
if (~isfield(options, 'maxfun'))
    options.maxfun = maxfun;
end
options.maxfun = min(options.maxfun, options.maxfun_dim*options.maxdim);
if (~isfield(options, 'sequential'))
    options.sequential = sequential;
end
if (~isfield(options, 'debug'))
    options.debug = debug;
end
if (~isfield(options, 'chkfunval'))
    options.chkfunval = chkfunval;
end
if (~isfield(options, 'output_xhist'))
    options.output_xhist = output_xhist;
end
if (~isfield(options, 'output_nlchist'))
    options.output_nlchist = output_nlchist;
end
if (~isfield(options, 'thorough_test'))
    options.thorough_test = thorough_test;
end
if (~isfield(options, 'minip'))
    options.minip = minip;
end
if (~isfield(options, 'maxip'))
    options.maxip = maxip;
end
if (~isfield(options, 'strict'))
    options.strict = strict;
end

% Set eval_options
have_eval_options = isfield(options, 'eval_options') && isstruct(options.eval_options) && ~isempty(fieldnames(options.eval_options));
if ~have_eval_options
    options.eval_options = eval_options;
end

if have_eval_options
    eval_options = options.eval_options;

    noise.type = 'relative';
    noise.nature = 'normal';
    noise.level = 0;
    if isfield(eval_options, 'noise') && isnumeric(eval_options.noise) && isscalar(eval_options.noise)
        noise.level = abs(eval_options.noise);
    elseif isfield(eval_options, 'noise') && isstruct(eval_options.noise)
        noise = eval_options.noise;
        if ~isfield(noise, 'type')
            noise.type = 'relative';
        end
        if ~isfield(noise, 'nature')
            noise.nature = 'normal';
        end
        if ~isfield(noise, 'level')
            noise.level = 1e-3;  % The default noise level if `noise` is present in `eval_options`
        end
        noise.level = abs(noise.level);
    end
    eval_options.noise = noise;
    if eval_options.noise.level == 0
        eval_options = rmfield(eval_options, 'noise');
    end

    dnoise.type = 'relative';
    dnoise.level = 0;
    if isfield(eval_options, 'dnoise') && isnumeric(eval_options.dnoise) && isscalar(eval_options.dnoise)
        dnoise.level = abs(eval_options.dnoise);
    elseif isfield(eval_options, 'dnoise') && isstruct(eval_options.dnoise)
        dnoise = eval_options.dnoise;
        if ~isfield(dnoise, 'type')
            dnoise.type = 'relative';
        end
        if ~isfield(dnoise, 'level')
            dnoise.level = 1e-3;  % The default dnoise level if `dnoise` is present in `eval_options`
        end
        dnoise.level = abs(dnoise.level);
    end
    eval_options.dnoise = dnoise;
    if eval_options.dnoise.level == 0
        eval_options = rmfield(eval_options, 'dnoise');
    end

    if isfield(eval_options, 'signif1')
        eval_options.signif = 1;
    elseif isfield(eval_options, 'signif2')
        eval_options.signif = 2;
    elseif isfield(eval_options, 'signif3')
        eval_options.signif = 3;
    elseif isfield(eval_options, 'signif4')
        eval_options.signif = 4;
    elseif isfield(eval_options, 'signif5')
        eval_options.signif = 5;
    elseif isfield(eval_options, 'signif6')
        eval_options.signif = 6;
    end

    if isfield(eval_options, 'single')
        eval_options.single = true;
    end

    options.eval_options = eval_options;
end

eval_options = options.eval_options;

% Revise options.nr
noisy_eval = (isfield(eval_options, 'noise') && eval_options.noise.level > 0);
if ~(options.perm || options.randomizex0 > 0 || noisy_eval)
    options.nr = 1;
end

% Revise options.ctol and options.cpenalty
if isfield(eval_options, 'dnoise')
    options.ctol = max(options.ctol, eval_options.dnoise.level);
end
if isfield(eval_options, 'noise')
    options.ctol = max(options.ctol, eval_options.noise.level);
end
if isfield(eval_options, 'signif')
    options.ctol = max(options.ctol, 10^(-eval_options.signif));
end
if isfield(eval_options, 'single') && eval_options.single
    options.ctol = max(options.ctol, eps('single'));
end
if options.randomizex0 > 0
    options.ctol = max(options.ctol, 1e-8);
end
options.cpenalty = min(options.cpenalty, 1/options.ctol);

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function solv_options = setsolvopt(solv, n, options)

solv_options = struct();
solv_options.rhobeg = options.rhobeg;
solv_options.rhoend = options.rhoend;
solv_options.maxfun = min(options.maxfun_dim*n, options.maxfun);  % may differ from options.maxfun
solv_options.ftarget = options.ftarget;
solv_options.output_xhist = options.output_xhist;
solv_options.output_nlchist = options.output_nlchist;
solv_options.iprint = 0;
solv_options.quiet = true;
solv_options.debug = options.debug;
solv_options.chkfunval = options.chkfunval;
%solv_options.scale = true;
solv_options.ctol = options.ctol;

if (strcmpi(solv, 'fmincon'))
    solv_options = optimoptions('fmincon');
    solv_options.MaxFunctionEvaluations = min(options.maxfun_dim*n, options.maxfun);
    solv_options.MaxIterations = options.maxit;
    solv_options.ObjectiveLimit = options.ftarget;
    solv_options.OptimalityTolerance = options.rhoend;
    solv_options.StepTolerance = options.rhoend;
    solv_options.ConstraintTolerance = min(1e-6, options.rhoend);
end
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = evalf(f, x, options)

if isfield(options, 'noise')
    noise = options.noise;
    if isstruct(noise) && isfield(noise, 'level') && noise.level > 0
        seed = 0.3*sin(1e8*abs(f))+0.3*cos(1e8*norm(x,9)) + 0.3*sin(100*norm(x,1))*cos(100*norm(x,Inf)) + 0.1*cos(norm(x));
        rng(min(options.ir*ceil(abs(10e6*seed)), 2^31));  % rng accepts integers between 0 and 2^32 - 1.

        switch lower(noise.nature)
        case {'uniform', 'u'}
            r = 2*rand-1;
        otherwise
            r = randn;
        end

        switch lower(noise.type)
        case {'absolute', 'additive', 'add', 'a', '+'}
            f = f + noise.level*r;
        otherwise
            f = f * (1 + noise.level*r);
        end
    end
end

if isfield(options, 'dnoise')
    dnoise = options.dnoise;
    if isstruct(dnoise) && isfield(dnoise, 'level') && dnoise.level > 0
        phi0 = 0.6*cos(1e8*norm(x,9)) + 0.3*sin(100*norm(x,1))*cos(100*norm(x,Inf)) + 0.1*cos(norm(x));
        noisimul = phi0*(4*phi0^2-3);
        switch lower(dnoise.type)
        case {'absolute', 'additive', 'add', 'a', '+'}
            f = f + dnoise.level*noisimul;
        otherwise
            f = f * (1 + dnoise.level*noisimul);
        end
    end
end

if isfield(options, 'single') && isscalar(options.single) && islogical(options.single) && options.single
    f = double(single(f));
end

if (isfield(options, 'signif'))
    sig = min(max(1, options.signif), 16);
    sf = eval(mat2str(f, sig));
    r = sin(sin(sig) + sin(1e8*f) + sum(abs(sin(1e8*x))) + sin(length(x)));
    f = sf + (f-sf)*(r+1);   % This makes the truncation more "irregular".
end

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = evalfun(fun, x, options, ir)
if isfield(options, 'single') && isscalar(options.single) && islogical(options.single) && options.single
    f = fun(double(single(x)));
else
    f = fun(x);
end
options.ir = ir;
f = evalf(f, x, options);
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cineq, ceq] = evalcon(con, x, options, ir)
if isfield(options, 'single') && isscalar(options.single) && islogical(options.single) && options.single
    [cineq, ceq] = con(double(single(x)));
else
    [cineq, ceq] = con(x);
end
options.ir = ir;
afun = @(f) evalf(f, x, options);
cineq = arrayfun(afun, cineq);
ceq = arrayfun(afun, ceq);
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cstrv = get_cstrv(x, Aineq, bineq, Aeq, beq, lb, ub, nonlcon)
lb(isnan(lb)) = -inf; % Replace the NaN in lb with -inf
ub(isnan(ub)) = inf; % Replace the NaN in ub with inf
bineq(isnan(bineq)) = inf; % Replace the NaN in bineq with inf
if ~isempty(Aeq)
    nan_eq = isnan(sum(abs(Aeq), 2)) & isnan(beq); % NaN equality constraints
    Aeq = Aeq(~nan_eq, :); % Remove NaN equality constraints
    beq = beq(~nan_eq);
end
if isempty(lb)
    lb = -inf(size(x));
end
if isempty(ub)
    ub = inf(size(x));
end
rineq = [];
req = [];
if ~isempty(Aineq)
    rineq = Aineq*x-bineq;
end
if ~isempty(Aeq)
    req = Aeq*x-beq;
end
if ~isempty(nonlcon)
    [nlcineq, nlceq] = nonlcon(x);
else
    nlcineq = [];
    nlceq = [];
end
cstrv = max([0; rineq; abs(req); lb-x; x-ub; nlcineq; abs(nlceq)], [], 'includenan');
% max(X, [], 'includenan') returns NaN if X contains NaN, and maximum of X otherwise
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function blacklist = black_list(solver)
%BLACK_LIST returns a list of problems that will be skipped when testing solver.
% Unless otherwise specified, the problems listed below take too much time and make the test
% on GitHub Actions run overtime.

blacklist = {};

switch(lower(solver))
case 'uobyqa'
    % With the following blacklist, there is no unconstrained problem in CUTEst (as of 230130) with
    % dimension between 51 and 100.
    blacklist = [blacklist, { ...
        'ARGTRIGLS', ...
        'BA-L1LS', ...
        'BA-L1SPLS', ...
        'BROWNAL', ...
        'CHNROSNB', ...
        'CHNRSNBM', ...
        'DIAMON2DLS', ...
        'DIAMON3DLS', ...
        'DMN15102LS', ...
        'DMN15103LS', ...
        'DMN15332LS', ...
        'DMN15333LS', ...
        'DMN37142LS', ...
        'DMN37143LS', ...
        'ERRINROS', ...
        'ERRINRSM', ...
        'HYDC20LS', ...
        'LUKSAN11LS', ...
        'LUKSAN12LS', ...
        'LUKSAN13LS', ...
        'LUKSAN14LS', ...
        'LUKSAN15LS', ...
        'LUKSAN16LS', ...
        'LUKSAN17LS', ...
        'LUKSAN22LS', ...
        'LUKSAN21LS', ...
        'LRCOVTYPE', ...
        'MANCINO', ...
        'QING', ...
        'SENSORS', ...
        'TOINTGOR', ...
        'TOINTPSP', ...
        'VARDIM', ...
        }];
case 'newuoa'
    blacklist = [blacklist, { ...
        'ARGTRIGLS', ...
        'BROWNAL', ...
        'COATING', ...
        'DIAMON2DLS', ...
        'DIAMON3DLS', ...
        'DMN15102LS', ...
        'DMN15103LS', ...
        'DMN15332LS', ...
        'DMN15333LS', ...
        'DMN37142LS', ...
        'DMN37143LS', ...
        'HYDC20LS', ...
        'LRA9A', ...
        'LRCOVTYPE', ...
        'LUKSAN14LS', ...
        'LUKSAN17LS', ...
        'LUKSAN21LS', ...
        'LUKSAN22LS', ...
        'PENALTY2', ...
        'PENALTY3', ...
        'VARDIM', ...
        }];
case 'lincoa'
    % For the following problems, the classical lincoa encounters SEGFAULT.
    blacklist = [blacklist, {'3PK', 'LSNNODOC', 'MAKELA4', 'OET1', 'QPCBOEI2', 'QPNBOEI2', 'SIPOW3', 'SIPOW4', 'TFI2'}];
    blacklist = [blacklist, { ...
        'DALLASM', ...
        'TARGUS', ...
        }];
case 'cobyla'
    % The following problems were observed to take excessive time during tests GitHub Actions and
    % make the tests run overtime. Some of them may not be very time-consuming during a "plain"
    % test but become more challenging with some perturbations or variations. The excessive time may
    % be also a result of infinite cycling encountered by the classical version of cobyla.
    % The number following the problem is the time in seconds taken by cobyla in a "plain" test on
    % 230130, which tested all linearly and nonlinearly constrained problems with at most 100
    % variables and 10000 constraints. Bound-constrained or unconstrained problems were not tested.
    blacklist = [blacklist, { ...
        'AIRPORT', ...      % 73
        'BATCH', ...        % 20
        'CHANDHEQ', ...     % 17
        'CHEBYQADNE', ...   % 546
        'CHNRSBNE', ...     % 18
        'CHNRSNBMNE', ...   % 32
        'CORE1', ...        % 64
        'CRESC132', ...
        'CVXQP1', ...       % 54
        'DALLASS', ...      % 3 (it takes a long time on GitHub Actions)
        'DECONVBNE', ...
        'DIAMON2D', ...     % 1415
        'DIAMON3D', ...     % 3703
        'DMN15102', ...     % 887
        'DMN15103', ...     % 3205
        'DMN15332', ...     % 838
        'DMN15333', ...     % 1441
        'DMN37142', ...     % 857
        'DMN37143', ...     % 2406
        'DUAL1', ...        % 73
        'DUAL2', ...        % 30
        'ERRINRSMNE', ...   % 16
        'FBRAIN2', ...
        'FBRAIN2NE', ...
        'FBRAIN3', ...
        'FEEDLOC', ...
        'HAIFAM', ...       % 173
        'HIMMELBI', ...     % 100
        'HYDCAR20', ...     % 648
        'HYDCAR6', ...
        'KISSING2', ...
        'LAKES', ...        % 65
        'LEVYMONE', ...     % 15
        'LHAIFAM', ...
        'LINSPANH', ...     % 3 (it takes a long time on GitHub Actions)
        'LUKSAN11', ...
        'LUKSAN12', ...     % 563
        'LUKSAN13', ...     % 508
        'LUKSAN14', ...     % 23
        'LUKSAN15', ...     % 19
        'LUKSAN16', ...     % 17
        'LUKSAN17', ...     % 25
        'LUKSAN21', ...     % 13
        'LUKSAN22', ...     % 19
        'MANCINONE', ...
        'MSS1', ...         % 39
        'OET4', ...
        'OET5', ...
        'OET6', ...
        'OET7', ...
        'QINGNE', ...
        'QPCBLEND' , ...
        'SPANHYD', ...      % 15
        'SWOPF', ...        % 10
        'TAX13322', ...     % 5
        'TAXR13322', ...    % 5
        'TRO4X4', ...       % 30
        'VANDERM1', ...     % 72
        'VANDERM2', ...     % 72
        'VANDERM3', ...     % 76
        'VESUVIO', ...
        'VESUVIOU', ...
         }];
    % For the following problems, the classical cobyla encounters SEGFAULT.
    blacklist = [blacklist, {'LAKES', 'TENBARS2', 'TENBARS3',  'TENBARS4', 'VANDERM4'}];
end
return
