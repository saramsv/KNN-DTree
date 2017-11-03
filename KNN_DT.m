data = importdata('breast-cancer-wisconsin.csv');
%cleaning
cleaned_data = data;
for i =1: size(cleaned_data,1)
    if isnan(cleaned_data(i,7))
        cleaned_data(i,7) = 0;
    end
end

for i =1: size(cleaned_data,1)
    if cleaned_data(i,7)==0
        cleaned_data(i,7) = ceil(mean(cleaned_data(:,7)));
    end
end

cleaned_data = cleaned_data(:,2:11);

% cleaned_data = cleaned_data(randperm(size(cleaned_data,1)),:);%randomize the order
% cleaned_data2 = zscore(cleaned_data(:,1:9));
% cleaned_data = [cleaned_data2 cleaned_data(:,10)]
mx = max(cleaned_data);
% Calculate the mean of each column
ave = mean(cleaned_data);
% Calculate the standard deviation of each column
sigma = std(cleaned_data);
% size(cleaned_data);
%train_data = cleaned_data(1:600,:);%10,:)%
tvdata = cleaned_data(1:600,:);%10,:)%
%validate_data = cleaned_data(426:562,:);%430,:)%
test_data = cleaned_data(600:699,:);
K = [2,3,4];%,5,6,7,8,16,32];%[5, 9];%
% acc = [];
% fold = 10;
% aves = [];
% for i = 1:length(K)
%     ave_fold = 0;
%     for j = 1:fold
%         validate_data = tvdata((j-1)*60+1 : j*60 , :);
%         if j == 1
%             train_data = tvdata(j*60+1:600,:);
%         elseif j == 10
%             train_data = tvdata(1:540,:);
%         else
%             train_data = [tvdata(1:(j-1)*60, :); tvdata(j*60 +1:600, :)];
%         end
%         classes = k_nearest_neighbors(train_data, validate_data, K(i));
%         Accuracy = performance(validate_data(:,10),classes);
%         ave_fold = ave_fold + Accuracy;
%         acc = [acc ; [K(i), Accuracy]];
%     end
%     aves = [aves ave_fold/fold];
% end
% acc;
% [maxK k_best] = max(aves);
% plot(acc(:,1), acc(:,2), '.b')
% ylim([0 1.10])
%knnsearch
names = ["ClumpThickness";"UniformityofCellSize";"UniformityofCellShape";"MarginalAdhesion";"SingleEpithelialCellSize";"BareNuclei";"BlandChromatin";"NormalNucleoli";"Mitoses"];
%[mytree, root] = tree;
for i = 1:length(K)
    %K(i)
    [result_tree, name, real_data, classified_data, structure] = classify_by_DT(tvdata, K(i));
    %disp(result_tree.tostring);
    p = sum(real_data(:,10) ~= classified_data(:,10));
    error = p/size(real_data,1)
    structure;
    size(structure);
    [t_real_data, t_classified_data] = classify_test_data(test_data, structure);
    p = sum(t_real_data(:,10) ~= t_classified_data(:,10));
    test_error = p/size(t_real_data,1)
    %structure
end

function classes = k_nearest_neighbors(train_data, validate_data,k)
    classes = [];
    distances = [];
    train_num = size(train_data, 1);
    validate_num = size(validate_data, 1);
    for i = 1 :validate_num
        distances = [];
        for j = 1: train_num
            diff = validate_data(i, 1:9)-train_data(j, 1:9);
            dis = sqrt(diff*diff');
            distances = [distances ; [dis, train_data(j,10)]];
        end
        distances;
        [~,idx] = sort(distances(:,1));
        distances = distances(idx,:);
        distances = distances(1:k,:);
        num_cat1 = sum(distances(:,2)==2);
        num_cat2 = sum(distances(:,2)==4);
        if num_cat1 >= num_cat2
            categ = 2;
        else
            categ = 4;
        end
        classes = [classes; categ];
        
    end
end

function Accuracy = performance(actual, predicted)
    TN = sum(actual(:,1)==predicted(:,1) & predicted(:,1)==2);
    TP = sum(actual(:,1)==predicted(:,1) & predicted(:,1)==4);
    FP = sum(actual(:,1)== 2 & predicted(:,1)==4);
    FN = sum(actual(:,1)== 4 & predicted(:,1)==2);
    T = table([TN;FN],[FP;TP]);
    T.Properties.VariableNames = {'benign','malignant'};
    T.Properties.RowNames = {'benign','malignant'};
    Accuracy = (TN +TP)/(TN + TP + FN + FP);
    TPR = TP/(TP + FN);
    PPV = TP/(TP + FP);
    TNR = TN/(TN + FP);
    f_score= PPV * TPR /(PPV + TPR);
end

function [temp , root, real_data, classified_data, structure] = classify_by_DT(data, max_depth, names, features, outs, classified_data, real_data ,depth, structure)

    if nargin < 3
        names = ["ClumpThickness";"UniformityofCellSize";"UniformityofCellShape";"MarginalAdhesion";"SingleEpithelialCellSize";"BareNuclei";"BlandChromatin";"NormalNucleoli";"Mitoses"];
        features = data(:,1:size(data, 2)-1);
        outs = [];
        classified_data= [];
        real_data  = [];
        structure = [];
        depth = 1;
        %max_depth = max_depth;
    else
        depth = depth + 1;
    end
    num_features = size(features,2);
    
    impurity_measures = [];
    for i = 1: num_features
        if ~ismember(i,outs)
            for j = 2:9
                branch1 = data(data(:,i)<j,:);
                branch2 = data(data(:,i)>=j,:);
                if size(branch1,1)~=0 & size(branch2,1)~=0 
                    p = sum(branch1(:,10)==2);
                    p = p/size(branch1,1);
                    if p == 1 | p == 0
                        entropy = 0;
                    else
                        entropy = -p*log2(p)-(1-p)*log2(1-p);
                    end
                    gini = 2*p*(1-p);
                    misclassification_error = 1- max(p,1-p);
                    impurity_measures = [impurity_measures [i; j; entropy]];
                end
            end
        end
    end
    if size(impurity_measures,1)~=0
        [minimum col] = min(impurity_measures(3,:));
        impurity_measures(:,col);
        outs = [outs impurity_measures(1,col)];
        branch1 = data(data(:,impurity_measures(1,col))< impurity_measures(2,col),:);
        branch2 = data(data(:,impurity_measures(1,col))>=impurity_measures(2,col),:);
    end

    % leaf
    if size(data,1)<=10 ...
        | entropy < 0.01 ...
        |length(outs) == num_features ...
        | size(branch1,1) == size(data, 1) ...  
        | size(branch2,1) == size(data, 1) ...
        | depth > max_depth ...
        
        [temp, root] = tree(data);
        real_data = [real_data ; data];
        p1 = sum(data(:,10) == 2);
        p1 = p1/size(data,1);
        p2 = sum(data(:,10) == 4);
        p2 = p2/size(data,1);
        class = ones(1,size(data,1));
        if p1 > p2
            class = 2*class; 
        else
            class = 4*class; 
        end
        data(:,10) = class';
        classified_data = [classified_data ; data];
        
        structure = [structure, [0;0]];
        
        return;
    else
        c = strcat(names(impurity_measures(1,col)),'>=');
        c = cellstr(strcat(c, num2str(impurity_measures(2,col))));
        [temp, root]= tree(c);
        structure = [structure, impurity_measures(1:2, col)];

        [child1, root1, real_data, classified_data, structure]= classify_by_DT(branch1, max_depth, names, features, outs, classified_data, real_data , depth, structure);
        [child2, root2, real_data, classified_data, structure] = classify_by_DT(branch2, max_depth, names, features, outs, classified_data, real_data , depth, structure);
        temp = temp.graft(root, child1);
        temp = temp.graft(root, child2);
    end
end
 
function [t_real_data, t_classified_data, col] = classify_test_data(data, structure, t_real_data, t_classified_data,col)
    if nargin < 3
        col = 1;
        t_real_data = [];
        t_classified_data = [];
    else
        col = col + 1;
    end
    if structure(1,col) == structure(2,col) & structure(1,col)==0
        t_real_data = [t_real_data ; data];
        p1 = sum(data(:,10) == 2);
        p1 = p1/size(data,1);
        p2 = sum(data(:,10) == 4);
        p2 = p2/size(data,1);
        class = ones(1,size(data,1));
        if p1 > p2
            class = 2*class; 
        else
            class = 4*class; 
        end
        data(:,10) = class';
        t_classified_data = [t_classified_data ; data];
        return
    end
    branch1 = data(data(:,structure(1,col))< structure(2,col),:);
    branch2 = data(data(:,structure(1,col))>=structure(2,col),:);
    col;
    [t_real_data, t_classified_data, col] = classify_test_data(branch1, structure, t_real_data, t_classified_data,col);
    [t_real_data, t_classified_data, col] = classify_test_data(branch2, structure, t_real_data, t_classified_data,col);    

end
    % MOA::THE FOLLOWING LINE IS THE ONLY LINE THAT NEEDS TO CHANGE
%     if size(data,1)==size(branch1,1) | size(branch2,1) <= 10
%         p1 = sum(branch1(:,10)==2);
%         p1 = p1/size(branch1,1);
%         p2 = sum(branch1(:,10)==4);
%         p2 = p2/size(branch1,1);
%         class = ones(1,size(branch1,1));
%         if p1 > p2
%             class = 2*class; 
%         else
%             class = 4*class; 
%         end
%         branch1(:,10) = class';
%         result = [result branch1];
%         [temp , root] = tree;
%         temp = temp.addnode(root, branch1)
%         return;
%     end
%     
%     if size(data,1)==size(branch2,1) |size(branch1,1) <= 10
%         p1 = sum(branch2(:,10)==2);
%         p1 = p1/size(branch2,1);
%         p2 = sum(branch2(:,10)==4);
%         p2 = p2/size(branch2,1);
%         class = ones(1,size(branch2,1));
%         if p1 > p2
%             class = 2*class; 
%         else
%             class = 4*class; 
%         end
%         branch2(:,10) = class';
%         result = [result branch2];
%         [temp root] = tree;
%         temp = temp.addnode(root, branch2)
%         return;
%     end
    

    %c = strcat(names(impurity_measures(1,col)),'>=');
    %c = cellstr(strcat(c, num2str(impurity_measures(2,col))));
    
    % Dar calle aval
     %[temp, root]= tree(c)
%     [temp, child_target1] = temp.addnode(1, branch1);
%     
%     % dar calle dovom
%     temp = tree(c)
%     [temp, child_target2]= temp.addnode(1, branch2);
%     
%     
% 
%     mytree = mytree.graft(root, temp);

    % If it's a leaf
%     if impurity_measures(3,col) <=0.02 | length(outs) == num_features
%         return;
%     end
%     
%     if impurity_measures(3,col) > 0.02
%         [child1_target, root]= classify_by_DT(branch1, names, features, outs, result);
%         [child2_target, root] = classify_by_DT(branch2, names, features, outs, result);
%         temp  = temp.addnode(root, child1_target);
%         temp = temp.addnode(root, child2_target);
%     end

    
    %t.get(3)
            
    %end
%end
    
    
    



























