lambda = [0,1,2,5,10,20,50];
log_lambda = [log(0.9),log(1),log(2),log(5),log(10),log(20),log(50)];
infection_averted =[386145,401215,402393,403206,405582,405384,405169];
travel_cost = [196572213,197389518,197637565,201724792,209260624,215302070,229792389];
figure
yyaxis left
plot(lambda,infection_averted,'b--', ...
    'LineWidth',3);
ylim([380000 420000])
ylabel('Infections Averted','FontSize',16)
xlim([-0.5 55])
yyaxis right
plot(lambda,travel_cost,'r:',...
    'LineWidth',3);
ax = gca;
ax.FontSize = 16; 
title({'Objectives Based on Various \lambda'},'Fontsize',18)
xlabel('\lambda','FontSize',16)
ylabel('Transportation Cost (Minutes)','FontSize',16)
