clear all
clc
%收敛曲线绘制
%载入数据
Loss_normal_20 = load("D:/jupyter_notebook/minist/Loss_normal_20.mat");
Val_Loss_normal_20 = load("D:/jupyter_notebook/minist/Val_Loss_normal_20.mat");
Loss_cum_20 = load("D:/jupyter_notebook/minist/Loss_cum_20.mat");
Val_Loss_cum_20 = load("D:/jupyter_notebook/minist/Val_Loss_cum_20.mat");
figure(9)
colormap(gray)
x = 1:20;
set(gca,'linewidth',0.5,'fontsize',10,'fontname','Times');
%依次设置坐标轴的属性分别为：线宽（4），刻度字号大小（30），刻度字体（罗马体）。
H = plot(x,Loss_normal_20.Loss_norma_20,'k--',x,Loss_cum_20.Loss_cum_20,'k');
legend(H,'DEA','IDEA');
xlabel('epochs','fontsize',10);
ylabel('Loss','fontsize',10)
