function [ell_hat_c, ell_hat_s, ell_star] = policy_functions(params,p_grid)

opcje = optimset('Display','off');

betta = params.betta;
dlta = params.dlta;
psi_vector = params.psi_vector;


Np = length(p_grid);


ell_hat_c = 0.5*ones(Np,   length(psi_vector));
ell_hat_s = 0.5*ones(Np,   length(psi_vector));
ell_star  = 0.5*ones(Np,Np,length(psi_vector)); 


% recover functions
% ----------------------------------
uu          = params.uu;
gg          = params.gg;
hh          = params.hh;
kk          = params.kk;
vv          = params.vv;
ff          = params.ff;
uu_prime    = params.uu_prime;
gg_prime    = params.gg_prime;
hh_prime    = params.hh_prime;
kk_prime    = params.kk_prime;
vv_prime    = params.vv_prime;
ff_prime    = params.ff_prime;
lbda        = params.lbda;
% ----------------------------------



%% optimal allocation
% -----------------------------------------------------
niter    = 1.0;
ell_star1 = ell_star;
while niter < 100        
    for i = 1:Np
        for j = 1:Np

            Lp = [p_grid(i);p_grid(j)];
            ell_guess = [ell_star(i,j,1);ell_star(i,j,2)];
            ell_sol   = fsolve(@planner_residual,ell_guess,opcje);
            ell_star1(i,j,1) = ell_sol(1);
            ell_star1(i,j,2) = ell_sol(2);
            
        end
    end
    distance = norm(ell_star1(:) - ell_star(:));
    disp([niter, distance])
    if distance < 0.000001
        break;
    end
    niter = niter + 1;
    ell_star = ell_star1;
end
% -----------------------------------------------------




%% non-cooperative equilibrium
% -----------------------------------------------------
niter    = 1.0;
ell_hat_c1 = ell_hat_c;
ell_hat_s1 = ell_hat_s;
ell_hat_s_prime = ell_hat_s;
while niter < 1000    
    ell_hat_c_South = ell_hat_c;    
    ell_hat_s_South = ell_hat_s;  
    for i = 1:Np
        for j = 1:length(psi_vector)
            ell_hat_s_prime(i,j) = Df(@ellhats,p_grid(i));
        end
    end
    for i = 1:Np
        for j = 1:length(psi_vector)
            Lp = p_grid(i);            
            ell_hat_c1(i,j) = fsolve(@cournot_residual,ell_hat_c(i,j),opcje);                        
            ell_hat_s1(i,j) = fsolve(@stack_residual,ell_hat_s(i,j),opcje);                        
        end
    end
    distance_c = norm(ell_hat_c1(:) - ell_hat_c(:));
    distance_s = norm(ell_hat_s1(:) - ell_hat_s(:));
    distance = max(distance_c,distance_s);
    ell_hat_c = ell_hat_c1;
    ell_hat_s = ell_hat_s1;
    disp([niter, distance_c, distance_s])
    if distance < 0.0000001
        break;
    end
    niter = niter + 1;
end
% -----------------------------------------------------





    function yout = ellhats(p)
        yout = interp_linear_1D(p_grid,p,ell_hat_s(:,j));
    end


    function yout = planner_residual(ell_vec)

        cstar = sum(ell_vec)/2;
        lhs = uu_prime(cstar,params) - vv_prime(ell_vec,params);
        
        p = psi_vector .* gg(ell_vec,params) + kk(Lp,params);        
        ellp = [interp_linear_2D(p_grid,p_grid,p(1),p(2),ell_star(:,:,2));...
                interp_linear_2D(p_grid,p_grid,p(1),p(2),ell_star(:,:,1))];
            

        pp = psi_vector .*gg(ellp,params) + kk(p,params);        
        ellpp = [interp_linear_2D(p_grid,p_grid,pp(1),pp(2),ell_star(:,:,1));...
                 interp_linear_2D(p_grid,p_grid,pp(1),pp(2),ell_star(:,:,2))];
             
        cstar_pp = sum(ellpp)/2;
        lbda_pp = lbda(cstar_pp,ellpp,psi_vector,params);
        
        rhs = hh_prime(p,params) + betta*(1-dlta)*ff_prime(p,params)*params.VR + ...
            betta*dlta* kk_prime(p,params).* ( hh_prime(pp,params) + betta*(1-dlta)*ff_prime(pp,params)*params.VR ) + ...
            (betta*dlta)^2 + kk_prime(p,params).*kk_prime(pp,params).*lbda_pp;
        
        
        yout = lhs - rhs .* psi_vector .* gg_prime(ell_vec,params);
    end


    function yout = cournot_residual(ell)
        
        LHS   = ( uu_prime(ell,params) - vv_prime(ell,params) ) ;
        
        pN    = psi_vector(j) * gg(ell,params) + kk(Lp,params);        
        ellSp = interp_linear_1D(p_grid,pN,ell_hat_c_South(:,j));
        
        pSp   = psi_vector(j) * gg(ellSp,params) + kk(pN,params);
        
        pi_S_prime = kk_prime(pN,params);
        
        kk_prime_pSp =  kk_prime(pSp,params);        
        ellNpp  = interp_linear_1D(p_grid,pSp,ell_hat_c(:,j));
        lbdaNpp = lbda(ellNpp,ellNpp,psi_vector(j),params);
        
        RHS = hh_prime(pN,params) + betta*(1-dlta)*ff_prime(pN,params)*params.VR + ...
            (betta*dlta)^2 * pi_S_prime * kk_prime_pSp * lbdaNpp;
        
        yout = LHS - RHS * psi_vector(j) * gg_prime(ell,params);
        
    end


    function yout = stack_residual(ell)
        
        LHS   = ( uu_prime(ell,params) - vv_prime(ell,params) ) ;
        
        pN    = psi_vector(j) * gg(ell,params) + kk(Lp,params);        
        ellSp = interp_linear_1D(p_grid,pN,ell_hat_s_South(:,j));
        
        pSp   = psi_vector(j) * gg(ellSp,params) + kk(pN,params);
        
        pi_S_prime = kk_prime(pN,params) + psi_vector(j) * gg_prime(ellSp,params) * ...
            interp_linear_1D(p_grid,pN,ell_hat_s_prime(:,j));
        
        kk_prime_pSp =  kk_prime(pSp,params);        
        ellNpp  = interp_linear_1D(p_grid,pSp,ell_hat_s(:,j));
        lbdaNpp = lbda(ellNpp,ellNpp,psi_vector(j),params);
        
        RHS = hh_prime(pN,params) + betta*(1-dlta)*ff_prime(pN,params)*params.VR + ...
            (betta*dlta)^2 * pi_S_prime * kk_prime_pSp * lbdaNpp;
        
        yout = LHS - RHS * psi_vector(j) * gg_prime(ell,params);
        
    end




end