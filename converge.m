clear all
clc
%�������߻���
%��������
Loss_normal_20 = load("D:/jupyter_notebook/minist/Loss_normal_20.mat");
Val_Loss_normal_20 = load("D:/jupyter_notebook/minist/Val_Loss_normal_20.mat");
Loss_cum_20 = load("D:/jupyter_notebook/minist/Loss_cum_20.mat");
Val_Loss_cum_20 = load("D:/jupyter_notebook/minist/Val_Loss_cum_20.mat");
figure(9)
colormap(gray)
x = 1:20;
set(gca,'linewidth',0.5,'fontsize',10,'fontname','Times');
%������������������Էֱ�Ϊ���߿�4�����̶��ֺŴ�С��30�����̶����壨�����壩��
H = plot(x,Loss_normal_20.Loss_norma_20,'k--',x,Loss_cum_20.Loss_cum_20,'k');
legend(H,'DEA','IDEA');
xlabel('epochs','fontsize',10);
ylabel('Loss','fontsize',10)
