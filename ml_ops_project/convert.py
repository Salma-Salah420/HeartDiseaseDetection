import pandas as pd

df = pd.read_csv('ML.xls')
df.to_csv('ML.csv', index=False)

print("✅ تم التحويل بنجاح (الملف كان بصيغة CSV في الأصل)")
