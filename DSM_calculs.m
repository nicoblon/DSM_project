close all; clear; clc
DSM_constants;

%----------------------------------------------------------
% Calculs pour le projet à partir de cette ligne

m_r = (pi*D_r*D_r*L_r/4 - pi*D_i*D_i*L_r/4 - pi*D_a*D_a*L_a/4)*rho_r + pi*D_a*D_a*L_a/4*rho_a; % masse totale

m_a = pi*(D_a*D_a - D_i*D_i)/4*L_a*rho_a; % masse aimant

m_r1 = pi*(D_r*D_r - D_a*D_a)/4*L_a*rho_r; % masse autour aimant

m_r2 = pi*(D_r*D_r - D_i*D_i)/4*(L_r - L_a) * rho_r; % reste de la masse

z_CG = (m_a*L_a/2 + m_r1*L_a/2 + m_r2*(3*L_r/4))/m_r; % depuis le point le plus à gauche

z_CGcentre = z_CG - L_r/2; % centre de masse sur z par rapport au centre

y_CGcentre = 0; % centre de masse sur y par rapport au centre

z_support1CG = -(z_CG - z_sup1); % position du support 1 par rapport au CG

z_support2CG = z_sup2 - z_CG; % position du support 2 par rapport au CG

z_balourdCG = z_CGcentre + L_a/2;

% Moment d'inertie y rotor 

    % découpage du rotor en trois cylindre de même densité. 
    % part 1: arbre de longueur L_r - L_a
    % part 2: arbre de longueur L_a
    % part 3: aimant de longueur L_a

    % Masses des différentes partie:
    M_1 = (pi/4)*rho_r*(D_r^2 - D_i^2)*(L_r - L_a);
    M_2 = (pi/4)*rho_r*(D_r^2 - D_a^2)*L_a;
    M_3 = (pi/4)*rho_a*(D_a^2 - D_i^2)*L_a;

    % formule pour un cylindre creux autour de leur centre de gravité: 
    % J_y = J_x = (1/12)*M*(3*R_i + 3*R_e + l^2)

    J_y1cg = (1/12)*M_1*(3*D_r/2 + 3*D_i/2 + (L_r - L_a)^2);
    J_y2cg = (1/12)*M_2*(3*D_r/2 + 3*D_a/2 + L_a^2);
    J_y3cg = (1/12)*M_3*(3*D_a/2 + 3*D_i/2 + L_a^2);

    % calcul des inertie au centre de masse du rotor:

    
    

Jrz = pi/32*(rho_r*(L_r - L_a)*(D_r^4 - D_i^4) + rho_a*L_a*(D_a^4 - D_i^4) + rho_r*L_a*(D_r^4 - D_a^4));
k_lx = E_l*L_l*h_l/t_l;
k_lz = E_l*L_l*t_l/h_l;

k_c1 = E_c * h_c * t_c / l_c1;
k_c2 = E_c * h_c * t_c / l_c2;
k_c3 = E_c * h_c * t_c / l_c3;
k_c4 = E_c * h_c * t_c / l_c4;
k_c5 = E_c * h_c * t_c / l_c5;
k_c6 = E_c * h_c * t_c / l_c6;
k_c7 = E_c * h_c * t_c / l_c7;


    J_y1 = J_y1cg + M_1*(L_a/2 + 0.5*L_r - z_CG)^2;
    J_y2 = J_y2cg + M_2*(z_CG - L_r/2)^2;
    J_y3 = J_y3cg + M_3*(z_CG - L_r/2)^2;

    J_ry = J_y1 + J_y2 + J_y3 % Inertie totale du rotor autour de y
