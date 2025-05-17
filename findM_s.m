function [m_s,boolWorks] = findM_s(plageBalourd, etendue_mesure, m_r, L_s1, L_s2, kl_x, J_rz_red, d, omega, m_s)
%FINDM_S Summary of this function goes here
%   Detailed explanation goes here

    M = [(m_r + 2*m_s) m_s*(L_s2-L_s1); m_s*(L_s2-L_s1) (J_rz_red + 2*m_s*(L_s1^2 + L_s2^2))];

    K = [4*kl_x 2*kl_x*(L_s2 - L_s1); 2*kl_x*(L_s2 - L_s1) 2*kl_x*(L_s1^2 + L_s2^2)];

    D = [1; d];

    Fcentrifuge_min = plageBalourd(1)*omega^2;
    Fcentrifuge_max = plageBalourd(2)*omega^2;

    invertedMatrix = inv(K-omega^2*M);

    accel_minVect = omega^4 * plageBalourd(1) * (invertedMatrix * D);
    accel_maxVect = omega^4 * plageBalourd(2) * (invertedMatrix * D);
    
    accel_min_x = accel_minVect(1)
    accel_max_x = accel_maxVect(1)
    accel_min_theta = accel_minVect(2)
    accel_min_theta = accel_maxVect(2)

    % accel_minVect est un vecteur -> on compare quelle accélération avec
    % les étendues de mesure 

    % on prend les accel en direction de x en x1 et x2?
    
    

    if(accel_max_x <= etendue_mesure(2) && accel_min_x >= etendue_mesure(1) && accel_max_theta <= etendue_mesure(2) && accel_min_theta >= etendue_mesure(1) )
        boolWorks = true;
    else
        boolWorks = false;
    end

end

