import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import torchvision.models as models
print(torch.version.cuda)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T1 = torch.Tensor([[1,2],[3,4], [5,6]])
T2 = torch.Tensor([[2,3],[4,5], [6,7]])
T  = torch.cat((T1,T2))
print(T)
T3 =torch.Tensor([  [8.4361e-01, 2.0286e-03, 4.9800e-03,1.9486e-03, 9.7893e-03,1.0837e-03],
                    [9.9918e-01, 2.4990e-05, 1.2347e-04,1.0123e-04, 6.5928e-05,4.0896e-05]
        ])

T4 = torch.argmax(T3,dim=1)

m = torch.zeros (T3.shape).scatter (1, T4.unsqueeze (1), 1.0)
print(m)
T5 = torch.argmax(T3,dim=1).unsqueeze(-1)
print(T4)

#print(confusion_matrix(T4,T5,normalize = None))
#print(np.sum(T3.numpy()))

#print(classification_report(T4, T5,target_names=['0','1']))
model = models.vgg19(pretrained=False).to(device)

