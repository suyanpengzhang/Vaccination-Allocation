%lambda = [0,1,2,5,10,20,50];
lambda = [0,150,300,500,800,1000];
infection_averted =[402361,405522,400373,399375,394000,369732];
travel_cost = [1115427,464405,62570,36184,9004,0];
figure
yyaxis left
plot(lambda,infection_averted,'o', ...
    'LineWidth',3,'MarkerSize',10);
ylim([360000 420000])
ylabel('Infections Averted','FontSize',16)
%xlim([-0.5 55])
yyaxis right
plot(lambda,travel_cost,'+',...
    'LineWidth',3,'MarkerSize',10);
ax = gca;
ax.FontSize = 16; 
title({'Objectives Based on Various \lambda'''},'Fontsize',18)
xlabel('\lambda''','FontSize',16)
ylabel('Inequity Score','FontSize',16)
