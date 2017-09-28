%%
% Info: Read raw data
% Auth: Shuan
% Env : MatLab2016b and upper
%%
function Y = RGB2YCBCR(Raw_Matrix, Width, Height)
    Raw_Matrix = uint8(Raw_Matrix);
    number = size(Raw_Matrix,1);
    stride = Width*Height;
    Y = zeros(number, stride);
    for i = 1:number
        R = Raw_Matrix(i,1:stride);
        G = Raw_Matrix(i,stride+1:stride*2);
        B = Raw_Matrix(i,stride*2+1:end);
        IMG(:,:,1) = reshape(R, Height, Width)';
        IMG(:,:,2) = reshape(G, Height, Width)';
        IMG(:,:,3) = reshape(B, Height, Width)';
        YCBCR = rgb2ycbcr(IMG);
        y = YCBCR(:,:,1);
        Y(i,:) = y(:);
        imshow(IMG);
    end
end
