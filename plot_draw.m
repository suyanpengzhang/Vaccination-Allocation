%lambda = [0,1,2,5,10,20,50];
lambda = [0,50,100,150,200,250,300];
infection_averted =[402361,402321,403865,407553,403844,403838,403719];
travel_cost = [1.96,1.935,1.48,1.166,0.213,0.104,0];
figure
yyaxis left
plot(lambda,infection_averted,'o', ...
    'LineWidth',3,'MarkerSize',10);
ylim([400000 410000])
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
