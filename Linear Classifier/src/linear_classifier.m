%%
% Auth: Shuan
% Info: Linear Classifier
%%
clc;
clear;
close all;
%%
max_iter = 1e2;
a = zeros(3,1);
c = 1;
a(3,1) = c;
lr = 1e-3;
epsilon = 1e-8;
err_1 = zeros(max_iter,1);
err_2 = zeros(max_iter,1);
[w1,w2,w3,w4] = Get_data();
b = 3*ones(size(w1,1),1);
%% Batch Update
% w1=w3;
% for iter = 1:max_iter
%     sum_y = zeros(3,1);
%     [y1,err_1(iter,1)] = verify_1(w1,a);
%     [y2,err_2(iter,1)] = verify_1(-1*w2,a);
%     if ~isempty(y1)
%         sum_y(:) = sum(w1(y1,:))';
%     end
%     if ~isempty(y2)
%         sum_y(:) = sum_y(:) + sum(-1*w2(y2,:))';
%     end
%     a = a + lr*sum_y;
%     a(3,1) = c;
% %     if abs(lr*sum_y) < epsilon
% %         disp('Stop early.');
% %         return;
% %     end
% end
%% Widrow-Hoff
% w1=-1*w4;
% for iter = 1:max_iter
%     [y1,err_1(iter,1)] = verify_2(w1, a, b);
%     [y2,err_2(iter,1)] = verify_2(w2, a, b);
%     for i = 1:size(w1,1)
%         a = a + lr*(b(i,1)-w1(i,:)*a)*w1(i,:)'/iter;
%         a(3,1) = c;
%     end
%     for i = 1:size(w2,1)
%         a = a + lr*(b(i,1)-w2(i,:)*a)*w2(i,:)'/iter;
%         a(3,1) = c;
%     end
% end
%% Ho-Kashyap
% Y = [w1;-w3];
% b = 1*ones(2*size(w1,1),1);
% for iter = 1:max_iter
%     e = Y*a - b;
%     e_abs = (e+abs(e))/2;
%     b = b + 2*lr*e_abs;
%     a = Y\b;
% %     if sum(abs(e)) < epsilon
% %         disp('Early Stop.');
% %         return;
% %     end
%     err_1(iter,1) = mean(abs(e));
%     err_2(iter,1) = min(e);
% end
%% Kelser
%% Train
N = 8;
Cls = 4;
Y = zeros(N*Cls*(Cls-1),3*Cls);
a = ones(3*Cls,1);
count = 1;
for i = 1:N
    Y(count,1:3) = w1(i,:);
    Y(count,4:6) = -w1(i,:);
    count = count + 1;
    Y(count,1:3) = w1(i,:);
    Y(count,7:9) = -w1(i,:);
    count = count + 1;
    Y(count,1:3) = w1(i,:);
    Y(count,10:12) = -w1(i,:);
    count = count + 1;
end
for i = 1:N
    Y(count,4:6) = w2(i,:);
    Y(count,1:3) = -w2(i,:);
    count = count + 1;
    Y(count,4:6) = w2(i,:);
    Y(count,7:9) = -w2(i,:);
    count = count + 1;
    Y(count,4:6) = w2(i,:);
    Y(count,10:12) = -w2(i,:);
    count = count + 1;
end
for i = 1:N
    Y(count,7:9) = w3(i,:);
    Y(count,1:3) = -w3(i,:);
    count = count + 1;
    Y(count,7:9) = w3(i,:);
    Y(count,4:6) = -w3(i,:);
    count = count + 1;
    Y(count,7:9) = w3(i,:);
    Y(count,10:12) = -w3(i,:);
    count = count + 1;
end
for i = 1:N
    Y(count,10:12) = w4(i,:);
    Y(count,1:3) = -w4(i,:);
    count = count + 1;
    Y(count,10:12) = w4(i,:);
    Y(count,4:6) = -w4(i,:);
    count = count + 1;
    Y(count,10:12) = w4(i,:);
    Y(count,7:9) = -w4(i,:);
    count = count + 1;
end
for iter = 1:max_iter
    sum_y = zeros(3*Cls,1);
    [y,err_1(iter,1)] = verify_1(Y,a);
    err_1(iter,1) = 1 - err_1(iter,1)/(N*Cls*(Cls-1));
    if ~isempty(y)
        sum_y(:) = sum(Y(y,:))';
    end
    a = a + lr*sum_y/iter;
%     a(3,1) = c;
%     a(6,1) = c;
%     a(9,1) = c;
%     a(12,1) = c;
%% Test
    a1 = a(1:3,1);
    a2 = a(4:6,1);
    a3 = a(7:9,1);
    a4 = a(10:12,1);
    correct = 0;
    Y_test = [w1(9:10,:);w2(9:10,:);w3(9:10,:);w4(9:10,:)];
    W = [a1,a2,a3,a4];
    Result = Y_test*W;
    [max_a,index] = max(Result,[],2);
    index = index - [1,1,2,2,3,3,4,4]';
    fprintf('ACC : %f\n', length(find(index==0))/8);
    err_2(iter,1) = length(find(index==0))/8;
end

%% Show Distribution
% figure(1);
% scatter(w2(:,1), w2(:,2), 'r');
% hold on;
% scatter(w4(:,1), w4(:,2), 'b');
% t = legend('w2','w4');
% t.FontSize = 15;
% t = title('Distribution of w2 and w4');
% t.FontSize = 15;
%% Batch Update
% figure(2);
% plot(err_1,'b-','LineWidth',2);
% hold on;
% plot(err_2,'r','LineWidth',2);
% t = legend('w3','w2');
% t.FontSize = 15;
% t = title('Misclassification sample number along with iteration');
% t.FontSize = 15;
% xlim([1,15]);
%% Widrow-Hoff
% figure(3);
% plot(err_1,'b-','LineWidth',2);
% hold on;
% plot(err_2,'r--','LineWidth',2);
% hold on;
% err_3 = err_2+err_1;
% plot(err_3,'g-','LineWidth',2);
% t = legend('w4 error','w2 error','overall error');
% t.FontSize = 15;
% t = title('Training error along with iteration.(b=3)');
% t.FontSize = 15;
% xlim([1,25]);
%% Ho-Kashyap
% figure(4);
% plot(err_1,'b-','LineWidth',2);
% hold on;
% plot(err_2,'r--','LineWidth',2);
% t = legend('traning loss','min(e)');
% t.FontSize = 15;
% t = title('Training error along with iteration.');
% t.FontSize = 15;
%% Kelser
figure(5);
plot(err_1,'b-','LineWidth',2);
hold on;
plot(err_2,'r-','LineWidth',2);
t = legend('Train ACC','Test ACC');
t.FontSize = 15;
t = title('Classification Accuracy Curve.');
t.FontSize = 15;



