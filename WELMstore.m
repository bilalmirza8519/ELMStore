function [Rn,Rp,Qn,Qp,nRn,nRp] = WELMstore(Pn,Tn, Elm_Type, nHiddenNeurons, ActivationFunction, IW, Bias,Rn,Rp,Qn,Qp,cnegold,cposold)

% Usage: OSELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction, N0, Block)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = OSELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction, N0, Block)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% nHiddenNeurons        - Number of hidden neurons assigned to the OSELM
% ActivationFunction    - Type of activation function:
%                           'rbf' for radial basis function, G(a,b,x) = exp(-b||x-a||^2)
%                           'sig' for sigmoidal function, G(a,b,x) = 1/(1+exp(-(ax+b)))
%                           'sin' for sine function, G(a,b,x) = sin(ax+b)
%                           'hardlim' for hardlim function, G(a,b,x) = hardlim(ax+b)
% N0                    - Number of initial training data used in the initial phase of OSLEM, which is not less than the number of hidden neurons
% Block                 - Size of block of data learned by OSELM in each step
%
% Output: 
% TrainingTime          - Time (seconds) spent on training OSELM
% TestingTime           - Time (seconds) spent on predicting all testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classifcation
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classifcation
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: OSELM('mpg_train', 'mpg_test', 0, 25, 'rbf', 75, 1);
% Sample2 classification: OSELM('segment_train', 'segment_test', 1, 180, 'sig', 280, 20);

    %%%%    Authors:    
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      
    %%%%    WEBSITE:    
    %%%%    DATE:       

%%%%%%%%%%% Macro definition

%  n=sum(N0N==-1);
%  p=sum(N0P==1);

 %n=sum(N0==-1);
% p=sum(N0==1);

%  P0=vertcat(P0N,P0P);
%  N0=vertcat(N0N,N0P);


 x=find(Tn==-1) ;
             % nn0=sum(T0==-1);
                   
             T0N=zeros(size(x));
             P0N=zeros(size(x), size(Pn,2));
             for f=1:size(x)
              T0N(f,:)=Tn(x(f,:),:);
              P0N(f,:)=Pn(x(f,:),:);
             end


 y=find(Tn==1) ;
             % nn0=sum(T0==-1);
                   
             T0P=zeros(size(y));
             P0P=zeros(size(y), size(Pn,2));
             for f=1:size(y)
              T0P(f,:)=Tn(y(f,:),:);
              P0P(f,:)=Pn(y(f,:),:);
             end

%  P0=vertcat(P0N,P0P);
%  N0=vertcat(T0N,T0P);
 
     cpos=sum(Tn==1);
cneg=sum(Tn==-1);
nRp=cpos+cposold;
nRn=cneg+cnegold;
% bil= ones(1,cneg)*1;
% mir= ones(1,cpos)*(cnegtot/cpostot);
% bilmir= [bil mir];
% W=diag(bilmir); 


% clear train_data test_data;
% 
 nTrainingData=size(P0N,1); 
 pTrainingData=size(P0P,1); 
% %nTestingData=size(TV.P,1);
% nInputNeurons=size(P0,2);

%%%%%%%%%%%% Preprocessing T in the case of CLASSIFICATION 
%if Elm_Type==CLASSIFICATION
    sorted_target=sort(T0N,1);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(nTrainingData)
        if sorted_target(i,1) ~= label(j,1)
            j=j+1;
            label(j,1) = sorted_target(i,1);
        end
    end
    nClass=2;
    nOutputNeurons=2;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(nTrainingData,nClass);
    for i = 1:nTrainingData
        for j = 1:nClass
            if label(j,1) == T0N(i,1)
                break; 
            end
        end
        temp_T(i,j)=1;
    end
    T0N=temp_T*2-1;
    
    
      sorted_target=sort(T0P,1);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(pTrainingData)
        if sorted_target(i,1) ~= label(j,1)
            j=j+1;
            label(j,1) = sorted_target(i,1);
        end
    end
    nClass=2;
    nOutputNeurons=2;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(pTrainingData,nClass);
    for i = 1:pTrainingData
        for j = 1:nClass
            if label(j,1) == T0P(i,1)
                break; 
            end
        end
        temp_T(i,j)=1;
    end
    T0P=temp_T*2-1;
    T0P(:,[1 2]) = T0P(:,[2 1])

    %%%%%%%%%% Processing the targets of testing
%     temp_TV_T=zeros(nTestingData,nClass);
%     for i = 1:nTestingData
%         for j = 1:nClass
%             if label(j,1) == TV.T(i,1)
%                 break; 
%             end
%         end
%         temp_TV_T(i,j)=1;
%     end
%     TV.T=temp_TV_T*2-1;
%end
%clear temp_T temp_TV_T sorted_target

%start_time_train=cputime;
%%%%%%%%%%% step 1 Initialization Phase
% P0=P(1:N0,:); 
% T0=T(1:N0,:);

IW0=IW;
%IW0 = rand(nHiddenNeurons,nInputNeurons)*2-1;
switch lower(ActivationFunction)
    case{'rbf'}
        Bias = rand(1,nHiddenNeurons);
%        Bias = rand(1,nHiddenNeurons)*1/3+1/11;     %%%%%%%%%%%%% for the cases of Image Segment and Satellite Image
%        Bias = rand(1,nHiddenNeurons)*1/20+1/60;    %%%%%%%%%%%%% for the case of DNA
        H0 = RBFun(P0,IW,Bias);
    case{'sig'}
       % Bias = rand(1,nHiddenNeurons)*2-1;
        Hn = SigActFun(P0N,IW0,Bias);
        Hp = SigActFun(P0P,IW0,Bias);
    case{'sin'}
        Bias = rand(1,nHiddenNeurons)*2-1;
        H0 = SinActFun(P0,IW,Bias);
    case{'hardlim'}
        Bias = rand(1,nHiddenNeurons)*2-1;
        H0 = HardlimActFun(P0,IW,Bias);
        H0 = double(H0);
end

tempHn=Hn'*Hn;
Rn=Rn+tempHn;
tempQn=Hn' * T0N;
Qn=Qn+tempQn;

tempHp=Hp'*Hp;
Rp=Rp+tempHp;
tempQp=Hp' * T0P;
Qp=Qp+tempQp;

% M = M - M * H' * (inv(W) + H * M * H')^(-1) * H * M; 
% beta = beta + M * H' *W* (N0 - H * beta);
% clear P0 N0 H0;