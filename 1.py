import streamlit as st
st.set_page_config(layout="wide")
st.title('机器学习梳理 ')
col1,col2=st.columns(2)
with col1:
    st.markdown("""
        <style>
            /* 将所有列表项的字体大小设为50px */
            ul {
                font-size: 50px;
            }
        </style>
        - 什么是机器学习？
        - 机器学习、神经网络以及深度学习到底有什么关系？
        - pytorch, anaconda, pycharm, cuda是什么？
        - 神经网络和torch.nn？
        - 激活函数是什么，怎么选？
        - 损失函数是什么？
        - 优化器是什么？
        - 卷积层、池化层到底是什么？
        - 常见算法？
    """, unsafe_allow_html=True)
with col2:
    st.image('v2-1.png' )
st.title('神经网络')
col1,col2=st.columns(2)
with col1:
    st.image('1.png')
    st.image('3.png',caption="神经元模型",)
with col2:
    st.image('2.png')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.image('4.png')
st.title('激活函数')
st.markdown("#### 神经元模型中的非线性函数就是我们常说的激活函数（激活函数active function=转移函数transfer function）")
active_function = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'Parametric ReLU', 'ELU', 'SeLU', 'Softmax', 'Swish', 'Maxout',
            'Softplus']
active_function_use = st.selectbox('请选择激活函数', active_function)
st.title('损失函数')
st.markdown("#### 衡量模型的输出估计y与真实的y之间的差距，给模型的优化指明方向(损失函数loss function=残差函数error function=代价函数cost function)")
loss_function = ['平均绝对误差损失 MAE', '均方差损失MSE', 'Huber', '分位数损失 Quantile Loss', '交叉熵损失 Cross Entropy Loss', '合页损失 Hinge Loss']
loss_function_use = st.selectbox('请选择损失函数', loss_function)
st.title('优化器')
st.markdown("#### 优化器在深度学习反向传播过程中，指引损失函数（目标函数）值不断逼近全局最小")
Optimizer = ['SGD', 'Momentum', 'NAG',' RMSprop','Adagrad','Adam', 'Adadelta', 'Nadam','AdamW',]
Optimizer_use = st.selectbox('请选择优化器', Optimizer)
st.title('常见算法')
st.markdown("""
 #### -常见的监督学习：线性回归、决策树、朴素贝叶斯分类、最小二乘法、逻辑回归、支持向量机、集成方法\n
#### -常见的无监督学习：聚类算法、主成分分析、奇异值分解、独立成分分析\n
#### -常用的分类器包括SVM、KNN、贝叶斯、线性回归、逻辑回归、决策树、随机森林、xgboost、GBDT、boosting、神经网络NN\n
#### -常见的降维方法包括TF-IDF、主题模型LDA、主成分分析PCA等等\n
#### -常用的回归技术：线性回归、多项式回归、逐步回归、岭回归、套索回归、ElasticNet回归\n"""
)
st.markdown("## 让我们利用pytorch开始训练模型吧")
col1,col2=st.columns([1,2])
with col1:
        st.markdown("- 1、准备数据集dataset\n"
                "- 2、加载数据集dataloader\n"
                "- 3、创建网络模型\n"
                "- 4、损失函数\n"
                "- 5、优化器\n"
                "- 6、设置训练参数\n"
                "- 7、进入训练\n"
                "- 8、进入测试\n"
                "- 9、保存模型\n")
with col2:
        with st.expander("准备数据集dataset"):
                st.code(
'from torch import nn\n'
'from torch.utils.data import DataLoader\n'
'train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),download=True)\n'            
'test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),download=True)\n')
        with st.expander("加载数据集"):
                st.code(
        'train_dataloader = DataLoader(train_data, batch_size=64)\n'
        'test_dataloader = DataLoader(test_data, batch_size=64)\n')
        with st.expander("创建网络模型"):
                st.code(
            '''class Test(nn.Module):
                def __init__(self):
                    super(Test, self).__init__()
                    self.model = nn.Sequential(
                        nn.Conv2d(3, 32, 5, 1, 2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 32, 5, 1, 2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 5, 1, 2),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(64*4*4, 64),
                        nn.Linear(64, 10)
                    )

                def forward(self, x):
                    x = self.model(x)
                    return x ''')
        with st.expander('损失函数'):
            st.code(
                'loss_fn=nn.CrossEntropyLoss()'
            )
        with st.expander('优化器'):
            st.code(
               '''learning_rate = 1e-2
                optimizer = torch.optim.SGD(test.parameters(), lr=learning_rate)'''
            )
        with st.expander('设置训练参数'):
            st.code(
        '''# 记录训练的次数
        total_train_step = 0
        # 记录测试的次数
        total_test_step = 0
        # 训练的轮数
        epoch = 100'''
            )
        with st.expander('进入训练'):
            st.code(
        ''' test.train()
            for data in train_dataloader:
                imgs, targets = data
                outputs = tudui(imgs)
                loss = loss_fn(outputs, targets)
                # 优化器优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_step = total_train_step + 1
                if total_train_step % 100 == 0:
                    print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))'''
            )
        with st.expander('进入测试'):
            st.code(
        '''test.eval()
            total_test_loss = 0
            total_accuracy = 0
            with torch.no_grad():
                for data in test_dataloader:
                    imgs, targets = data
                    outputs = tudui(imgs)
                    loss = loss_fn(outputs, targets)
                    total_test_loss = total_test_loss + loss.item()
                    accuracy = (outputs.argmax(1) == targets).sum()
                    total_accuracy = total_accuracy + accuracy
            print("整体测试集上的Loss: {}".format(total_test_loss))
            print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
            total_test_step = total_test_step + 1'''
            )
        with st.expander('保存模型'):
            st.code(
           'torch.save(test, "test_{}.pth".format(i))')







