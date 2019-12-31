function [n1, n2, n3, n4] = normcheck(A_bttb, A_circ, k)

n10 = 3*norm(k);
n11 = 3*norm(k.');

n1 = min(n10, n11);
n2 = normest(A_bttb,0.001);
n3 = normest(A_circ,0.001);
n4 = norm( squeeze(reshape(k,[9,1])), 1);

end