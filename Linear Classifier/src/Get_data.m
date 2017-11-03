function [w1, w2, w3, w4] = Get_data()
w1 = [ 0.1  1.1;...
       6.8  7.1;...
      -3.5 -4.1;...
       2.0  2.7;...
       4.1  2.8;...
       3.1  5.0;...
      -0.8 -1.3;...
       0.9  1.2;...
       5.0  6.4;...
       3.9  4.0];

w2 = [ 7.1  4.2;...
      -1.4 -4.3;...
       4.5  0.0;...
       6.3  1.6;...
       4.2  1.9;...
       1.4 -3.2;...
       2.4 -4.0;...
       2.5 -6.1;...
       8.4  3.7;...
       4.1 -2.2];

w3 = [-3.0 -2.9;...
      0.54  8.7;...
       2.9  2.1;...
      -0.1  5.2;...
      -4.0  2.2;...
      -1.3  3.7;...
      -3.4  6.2;...
      -4.1  3.4;...
      -5.1  1.6;...
       1.9  5.1];

w4 = [-2.0 -8.4;...
      -8.9  0.2;...
      -4.2 -7.7;...
      -8.5 -3.2;...
      -6.7 -4.0;...
      -0.5 -9.2;...
      -5.3 -6.7;...
      -8.7 -6.4;...
      -7.1 -9.7;...
      -8.0 -6.3];
w1(:,3) = 1;
w2(:,3) = 1;
w3(:,3) = 1;
w4(:,3) = 1;
end
