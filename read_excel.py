import pandas as pd

# 读取Excel文件
try:
    df = pd.read_excel('合并结果.xlsx')
    print("文件读取成功！")
    print(f"数据形状: {df.shape}")
    print("\n列名:")
    print(df.columns.tolist())
    print("\n前5行数据:")
    print(df.head())
    print("\n数据类型:")
    print(df.dtypes)
except Exception as e:
    print(f"读取文件时出错: {e}")
