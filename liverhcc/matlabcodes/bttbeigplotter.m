nx = 32;
ny = 32;
S = 10000;

true_circ_norm = zeros(S,1);
true_bttb_norm = zeros(S,1);
bound_norm_kernel = zeros(S,1);
bound_Holder = zeros(S,1);
c = zeros(S,1);

for s = 1:S

    [A_bttb, k] = bttbmaker(nx,ny);
    [A_circ, k]  = circulantmaker(nx,ny,k);
    [n1, n2, n3, n4] = normcheck(A_bttb, A_circ, k);
    
    bound_norm_kernel(s,1) = n1;
    true_bttb_norm(s,1) = n2;
    true_circ_norm(s,1) = n3;
    bound_Holder(s,1) = n4;
    
    if n1 < n4
        c(s,1) = 1;
    end
    
%     sigma = sort(svds(A, nx*ny, 'largest', 'MaxIterations', 100));
%     mmm = max(sigma);
%     ccc = min( 1, max( 1-(mmm./10), 0 ));
%     plot(sigma, 'Color', ccc.*ones(3,1));
end

figure;
hold on;
scatter(bound_norm_kernel, bound_Holder,2,c);
x = 0:0.01:16;
plot(x,x);
xlabel('3 || K ||_2');
ylabel('Holder bound');

figure;
hold on;
scatter(true_circ_norm, true_bttb_norm,2,c);
plot(x,x);
xlabel('|| A_{circ} ||_2');
ylabel('|| A_{bttb} ||_2');

figure;
hold on;
scatter(true_circ_norm, bound_Holder,2,c);
plot(x,x);
xlabel('|| A_{circ} ||_2');
ylabel('Holder bound');

figure;
hold on;
scatter(true_bttb_norm, bound_Holder,2,c);
plot(x,x);
xlabel('|| A_{bttb} ||_2');
ylabel('Holder bound');

figure;
hold on;
scatter(true_circ_norm, bound_norm_kernel,2,c);
plot(x,x);
xlabel('|| A_{circ} ||_2');
ylabel('3 || K ||_2');

figure;
hold on;
scatter(true_bttb_norm, bound_norm_kernel, 2,c);
plot(x,x);
xlabel('|| A_{bttb} ||_2');
ylabel('3 || K ||_2');