function [A, k] = bttbmaker(nx,ny)

k = randn(3);
A = sparse(nx*ny, nx*ny);
lb = [2,1,1];
ub = [ny,ny,ny-1];
for i=1:3
    
    % construct T_i
    Ti = zeros(nx,nx);
    for j=1:3
        Ti = Ti + diag(k(i,j).*ones(nx-abs(j-2),1),j-2);
    end
    
    % construct diag blocks of A
    for j=lb(i):ub(i)
        A((j-1)*nx+1:j*nx,(j-(3-i))*nx+1:(j-(2-i))*nx) = Ti;
    end
    
end

end