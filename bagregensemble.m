trainDF = csvread('trainDF.csv',1,1)
testDF = csvread('testDF.csv',1,1)
validateDF = csvread('validateDF.csv',1,1)
ens = fitensemble(trainDF(:,1:(size(trainDF,2)-1)), trainDF(:,(size(trainDF,2))), 'Bag', 500 ,'Tree', 'type','classification')
Yfit = predict(ens,validateDF(:,1:(size(validateDF,2)-1)));
csvwrite('result_bag',Yfit)
ens2 = fitensemble(trainDF(:,1:(size(trainDF,2)-1)), trainDF(:,(size(trainDF,2))), 'LSBoost', 500 ,'Tree')
Yfit2 = predict(ens2,validateDF(:,1:(size(validateDF,2)-1)));
csvwrite('result_boost',Yfit2)