% Arbre
rho_r = 8000;
D_r = 9 / 1000;
L_r = 90 / 1000;

% Aimant
rho_a = 7000;
D_a = 7 / 1000;
L_a = 45 / 1000;

% Rotor
D_i = 1.5 / 1000;

% Position des supports
z_sup1 = 0.1 * L_r;
z_sup2 = 0.9 * L_r;

% Lames 
E_l = 70 * 10^9;
L_l = 25 / 1000;
h_l = 7.5 / 1000;
t_l = 0.75 / 1000;

% Accéléromètre
f_coupure = 200; % Hz
etendue_mesure = [0.005 1];

% Moteur
D_m = 38 / 1000;
J_mz = 28 / 1000^2;

% Poulie
D_p = 20 / 1000;
J_pz = 0.20 / 1000^2;

% Courroie
E_c = 2 * 10^9;
h_c = 9 / 1000;
t_c = 0.8 /1000;
l_c1 = 280 / 1000;
l_c2 = 150 / 1000;
l_c3 = 200 / 1000;
l_c4 = 190 / 1000;
l_c5 = 120 / 1000;
l_c6 = 120 / 1000;
l_c7 = 300 / 1000;

l_cArray = [l_c1 l_c2 l_c3 l_c4 l_c5 l_c6 l_c7];

% Valeurs empiriques couple résistant rotor
T_resNulle = 1.4 / 1000;
T_resMax = 5.6 / 1000; 
