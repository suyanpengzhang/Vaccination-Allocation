%lambda = [0,1,2,5,10,20,50];
lambda = [0,50,100,150,200,250,300,500,1000];
infection_averted =[403588,403559,405491,408084,403847,403718,403718,403718,403718];
travel_cost = [1.96,1.935,1.365,0.831,0.197,0,0,0,0];
figure
yyaxis left
plot(lambda,infection_averted,'o-', ...
    'LineWidth',3,'MarkerSize',10);
ylim([400000 410000])
ylabel('Infections Averted','FontSize',16)
%xlim([-0.5 55])
yyaxis right
plot(lambda,travel_cost,'+-',...
    'LineWidth',3,'MarkerSize',10);
ax = gca;
ax.FontSize = 16; 
title({'Objectives Based on Various \lambda'''},'Fontsize',18)
xlabel('\lambda''','FontSize',16)
ylabel('Inequity Score','FontSize',16)
