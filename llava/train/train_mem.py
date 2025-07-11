
import os
import sys
sys.path.append("/data5/yunfei/LLaVA")
from llava.train.train import train
import inspect

if __name__ == "__main__":
    # 获取train函数的模块
    module = inspect.getmodule(train)

    # # 打印模块的文件路径
    # print(f"train 函数导入自: {module.__file__}")    
    # print(os.getenv("CONDA_DEFAULT_ENV"))
    # print(os.getenv("PYTHONPATH"))
    try:
        train()
    except Exception as e:
        print(e)
