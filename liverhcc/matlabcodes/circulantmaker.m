function [A, k] = circulantmaker(nx, ny, k)

nn = nx*ny;
A = sparse(nn, nn);
for i=1:3
    
    % construct T_i
    Ti = zeros(nx,nx);
    for j=1:3
        Ti = Ti + diag(k(i,j).*ones(nx-abs(j-2),1),j-2);
    end
    Ti(1,nx) = k(i,1);
    Ti(nx,1) = k(i,3);
    
    % construct diag blocks of A
    for j=1:ny
        A(mod((j-1)*nx:j*nx-1, nn)+1,mod((j-(3-i))*nx:(j-(2-i))*nx-1 ,nn)+1) = Ti;
    end
    
end


end