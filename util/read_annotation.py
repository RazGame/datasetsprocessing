import pandas as pd
#def read_lidc_annotation(dir):
file = r'annotation.xls'
df = pd.read_excel(file)
print ('PatientID',df['TCIA Patient ID'], 'Conclusion', df['Conclusion'])