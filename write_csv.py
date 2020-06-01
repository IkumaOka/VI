#coding:utf-8
import csv

f = open('data.csv', 'w')

csvWriter = csv.writer(f)

val = 0
for num in range(1, 5):
   listData = []
   val = num
   listData.append(val)
   for loop in range(0, 5):
      val = val * 10 + num
      listData.append(val)

   print(listData)
   csvWriter.writerow(listData)

f.close()