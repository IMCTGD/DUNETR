<font size=5>DUNETRsize=5</font>
基于Transformer和CNN双并行分支编码器神经网络的冠状动脉分割 </br>
 
## 使用说明
如果您打算使用这个项目，请按照以下步骤操作：</br>
 
1. 安装必要的依赖项:</br>
  - python=3.8.12
  - python_abi=3.8
  - pytorch=1.8.0
  - torchaudio=0.8.0
  - torchfile=0.1.0
  - torchvision=0.9.0
具体详见文件environment.yaml</br>

2. 运行项目:</br>
训练：my_train.py		运行方式 python my_train.py
验证：dicetest.py		运行方式 python dicetest.py		
测试：dicetest-test.py	运行方式 python dicetest-test.py
模型：DUNETR.py

