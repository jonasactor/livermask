nx = 16;
ny = 16;
S = 10000;

true_circ_norm = zeros(S,1);
true_bttb_norm = zeros(S,1);
bound_norm_kernel = zeros(S,1);
bound_Holder = zeros(S,1);

for s = 1:S

    [A_bttb, k] = bttbmaker(nx,ny);
    [A_circ, k]  = circulantmaker(nx,ny,k);
    [n1, n2, n3, n4] = normcheck(A_bttb, A_circ, k);
    
    bound_norm_kernel(s,1) = n1;
    true_bttb_norm(s,1) = n2;
    true_circ_norm(s,1) = n3;
    bound_Holder(s,1) = n4;
    
    
%     sigma = sort(svds(A, nx*ny, 'largest', 'MaxIterations', 100));
%     mmm = max(sigma);
%     ccc = min( 1, max( 1-(mmm./10), 0 ));
%     plot(sigma, 'Color', ccc.*ones(3,1));
end

hold on;
scatter(true_bttb_norm, bound_Holder,2);
x = 0:0.01:16;
plot(x,x);
