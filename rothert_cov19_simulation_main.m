clearvars
clc

global parametry

parametry.betta       = 0.90;
parametry.dlta        = 0.75;
parametry.allocations = {'Planner','Cournot','Stackelberg'};
parametry.T           = 20;
parametry.teta_g      = 0.50;
parametry.teta_v      = 0.50;
parametry.teta_h      = 0.50;
parametry.sgma        = 2.00;
parametry.kapa_const  = 0.90;
parametry.ff_const    = 0.10;
parametry.hh_const    = 5.00;
parametry.vv_const    = 0.50;

addpath 'C:\Users\R2D2\Dropbox\work\research\computing\library\matlab'

psi_vector = [0.15;0.18];
parametry.psi_vector = psi_vector;

parametry.uu        = @uu;
parametry.gg        = @gg;
parametry.hh        = @hh;
parametry.kk        = @kk;
parametry.vv        = @vv;
parametry.ff        = @ff;
parametry.uu_prime  = @uu_prime;
parametry.gg_prime  = @gg_prime;
parametry.hh_prime  = @hh_prime;
parametry.kk_prime  = @kk_prime;
parametry.vv_prime  = @vv_prime;
parametry.ff_prime  = @ff_prime;
parametry.lbda      = @lbda;

p_grid = linspace(0,0.30,20)';


%% post-pandemic VR
% --------------------------------------------------------------------------------------------------
low = 0.00;
hih = 10.0;
while hih - low > 0.0000001
    ell = (hih + low)/2;
    rhs = uu_prime(ell,parametry);
    lhs = vv_prime(ell,parametry);
    residual = rhs - lhs;
    if residual < 0
        hih = ell;
    elseif residual > 0
        low = ell;
    else
        break;
    end
end
parametry.VR = ( uu(ell,parametry) - vv(ell,parametry) ) / ( 1-parametry.betta );
% --------------------------------------------------------------------------------------------------
%%




%% policy functions
[ell_hat_C, ell_hat_S, ell_star] = policy_functions(parametry,p_grid);



%% simulation
psi_path  = [psi_vector'; flip(psi_vector')];
psi_path  = repmat(psi_path,parametry.T/2,1);
pp_path   = zeros(size(psi_path));
pp_path_C = pp_path;
pp_path_S = pp_path;
for t = 1:size(pp_path,1)-1

    % vector of psi's
	psi_vect = psi_path(t,:)';
    psi_low = find(abs(psi_vect - min(psi_vect)) < 0.0001);
    psi_hih = find(abs(psi_vect - max(psi_vect)) < 0.0001);

    % optimal allocation
    Lp = pp_path(t,:)';
    ell_vec = [interp_linear_2D(p_grid,p_grid,Lp(psi_low),Lp(psi_hih),ell_star(:,:,psi_low));interp_linear_2D(p_grid,p_grid,Lp(psi_low),Lp(psi_hih),ell_star(:,:,psi_hih))];
    p_vec   = psi_path(t,:)' .* gg(ell_vec,parametry) + kk(Lp,parametry);
    pp_path(t+1,:) = p_vec';
    
    % Cournot allocation
    Lp = pp_path_C(t,:)';
    ell_vec = [interp_linear_1D(p_grid,Lp(2),ell_hat_C(:,psi_low));interp_linear_1D(p_grid,Lp(1),ell_hat_C(:,psi_hih))];
    p_vec   = psi_path(t,:)' .* gg(ell_vec,parametry) + kk(Lp,parametry);
    pp_path_C(t+1,:) = p_vec';
    
    % Stack allocation
    Lp = pp_path_S(t,:)';
    ell_vec = [interp_linear_1D(p_grid,Lp(2),ell_hat_S(:,psi_low));interp_linear_1D(p_grid,Lp(1),ell_hat_S(:,psi_hih))];
    p_vec   = psi_path(t,:)' .* gg(ell_vec,parametry) + kk(Lp,parametry);
    pp_path_S(t+1,:) = p_vec';
    
end



ymin = 0.0;
ymax = max(pp_path(:,1));
ymax = max(ymax,max(pp_path_C(:,1)));
ymax = max(ymax,max(pp_path_S(:,1)));

colorVec = {'r', 'k'};
linVec = {'-', '-.'};

figure(1)
subplot(1,3,1);
for ip = 1:2
plot(pp_path(:,ip),'LineWidth',2,'Color',colorVec{ip},'LineStyle',linVec{ip}); 
hold on
end
ylabel('New Infections')
ylim([0.0,1.1*ymax])
set ( gca, 'FontSize',14 )
title('Planner')

subplot(1,3,2);
for ip = 1:2
plot(pp_path_C(:,ip),'LineWidth',2,'Color',colorVec{ip},'LineStyle',linVec{ip}); ylim([0.0,1.1*ymax])
hold on
end
xlabel('Time')
set ( gca, 'FontSize',14 )
title('Cournot')

subplot(1,3,3);
for ip = 1:2
plot(pp_path_S(:,ip),'LineWidth',2,'Color',colorVec{ip},'LineStyle',linVec{ip}); ylim([0.0,1.1*ymax])
hold on
end
set ( gca, 'FontSize',14 )
title('Stackelberg')
legend('North', 'South', 'Location','southeast')


plot2a = pp_path_C./pp_path;
plot2b = pp_path_S./pp_path_C;
figure(2)
subplot(1,2,1);
for ip = 1:2
plot(plot2a(:,ip),'LineWidth',2,'Color',colorVec{ip},'LineStyle',linVec{ip});
hold on
end
xlabel('Time')
ylabel({'New Infections';'(Planner = 1)'})
set ( gca, 'FontSize',14 )
title('Cournot relative to Planner')
subplot(1,2,2);
for ip = 1:2
plot(plot2b(:,ip),'LineWidth',2,'Color',colorVec{ip},'LineStyle',linVec{ip});
hold on
end
ylabel({'New Infections';'(Cournot = 1)'})
xlabel('Time')
set ( gca, 'FontSize',14 )
title('Stackelberg relative to Cournot')
legend('North', 'South', 'Location','northeast')








% g(ell)
function yout = gg(ell,params)
yout = ell.^(1+params.teta_g) ./ (1+params.teta_g);
end
function yout = gg_prime(ell,params)
yout = ell.^(params.teta_g);
end

% kappa(p)
function yout = kk(p,params)
yout = params.kapa_const*p;
end
function yout = kk_prime(p,params)
yout = params.kapa_const;
end

% h(p)
function yout = hh(p,params)
yout = params.hh_const * p.^(1+params.teta_h) ./ (1+params.teta_h);
end
function yout = hh_prime(p,params)
yout = params.hh_const * p.^(params.teta_h);
end


% f(p)
function yout = ff(p,params)
yout = params.ff_const*p;
end
function yout = ff_prime(p,params)
yout = params.ff_const;
end

% u(c)
function yout = uu(c,params)
if abs(params.sgma - 1) > 0.0001
yout = c.^(1-params.sgma) / (1-params.sgma);
else
yout = log(c);
end
end
function yout = uu_prime(c,params)
yout = c.^(-params.sgma);
end

% v(ell)
function yout = vv(ell,params)
yout = params.vv_const * ell.^(1+params.teta_v) ./ (1+params.teta_v);
end
function yout = vv_prime(ell,params)
yout = params.vv_const * ell.^(params.teta_v);
end

% lambda
function lbda_out = lbda(c,ell,ppsi,params)
lbda_out = (uu_prime(c,params) - vv_prime(ell,params)) ./ ppsi ./ gg_prime(ell,params);
end



