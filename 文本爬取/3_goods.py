import requests
import urllib
import time
from bs4 import BeautifulSoup
import sys
import os
# path = '货物/'
path = 'test1000/货物/'
if not os.path.isdir(path):
	os.mkdir(path)
# urls=['http://search.ccgp.gov.cn/bxsearch?searchtype=2&page_index={}&bidSort=&buyerName=&projectId=&pinMu=1'.format(x) for x in range(1,101)]
urls=['http://search.ccgp.gov.cn/bxsearch?searchtype=2&page_index={}&bidSort=&buyerName=&projectId=&pinMu=1&timeType=5'.format(x) for x in range(1,1000)]
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/51.0.2704.63 Safari/537.36'}
headers2 = {'User-Agent':"Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "}
# print(urls[0])
index = 0
def extract_content():
	global index 
	global path
	for url in urls:
		try:
			html=requests.get(url=url,headers=headers).content.decode('utf-8')
			soup = BeautifulSoup(html)
			items = soup.find_all('a',attrs={'style':"line-height:18px",'target':"_blank"})
			hrefs = []
			titles = []
			for item in items:
				hrefs.append(item['href'])
				titles.append(item.get_text().strip())
		except(Exception):
			continue
		if len(hrefs) != len(titles): continue
		for href,title in zip(hrefs,titles):
			try:
				print('开始采集第{}个网址样本'.format(index+1))
				# print(href)
				# time.sleep(1)
				html_1 = requests.get(url=href,headers=headers2).content.decode('utf-8')
				# print(html_1)
				# sys.exit()
				# print(title)
				# sys.exit()
				soup = BeautifulSoup(html_1)
				content = soup.find_all('div',attrs={'class':'vF_detail_content'})[0]
				content = title + content.get_text().strip() 
				with open(path+'{}.txt'.format(index),'w',encoding='utf-8')as f:
					f.write(content)
			except(Exception):
				# print('title中出现/')
				# name = str(index)+title.split('/')[0]
				# with open(path+'{}.txt'.format(name),'w',encoding='utf-8')as f:
				# 	f.write(content)
				# print('重新命名为',name)
				pass
			index += 1
extract_content()
