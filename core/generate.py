import re
import sys
import requests
from bs4 import BeautifulSoup
import xmnlp
import word2vec_Pagerank as wp
from collections import defaultdict
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import Qt
xmnlp.set_model("./xmnlp-onnx-models")

class MyWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.biography_text=''
        self.phases=[]
        self.text=''
        self.url=''
        self.passages=[]
        self.count=0


    def init_ui(self):
        self.ui = uic.loadUi("../ui/mainwindow.ui")
        self.ui.setWindowTitle('Resume_Generate_4_the_old')
        # print(self.ui.__dict__)  # 查看ui文件中有哪些控件

        # 提取要操作的控件
        #按钮
        self.start=self.ui.start
        self.open=self.ui.actionopen#打开文件

        #输出表
        self.source=self.ui.source
        self.result=self.ui.result
        self.t=self.ui.time
        self.p=self.ui.persons
        self.l=self.ui.location
        
        # 输入
        self.input=self.ui.input
        
        #按钮和操作
        self.start.clicked.connect(lambda:self.generate())
        self.open.triggered.connect(lambda:self.justread())

    def generate(self):
        # 初始化
        self.count+=1
        self.source.clear()
        self.result.clear()
        self.t.clear()
        self.p.clear()
        self.l.clear()
        self.biography_text=''
        self.phases=[]
        self.text=''
        self.url=''
        self.passages=[]


        # 生成步骤
        self.pachong()
        self.cleandata()
        self.writefile()
        self.readfile()
        self.NER()
        self.textrank()

    def justread(self):
        # 初始化     
        self.source.clear()
        self.result.clear()
        self.t.clear()
        self.p.clear()
        self.l.clear()
        self.biography_text=''
        self.phases=[]
        self.text=''
        self.url=''
        self.passages=[]

        filename=QFileDialog.getOpenFileName(self,"选择文件(路径中不能有中文)","../spider","(*.txt)")
        self.input.setText(filename[0])

        with open(filename[0], "r", encoding="utf-8") as fin: #F:\\Integrated_design\\test.txt
            self.text = fin.read()
        self.source.setText(self.text) 

        self.NER()
        self.textrank()


    # 1.爬虫
    def pachong(self):

        url =self.input.text()
        # print(url)
        response = requests.get(url)
        html = response.text

        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(html, 'html.parser')

        # 提取p标签
        self.biography_text = ' '.join([p.text for p in soup.find_all('p') if p.text.strip() != ''])

    #2.清洗数据
    def cleandata(self):        
        #去除水印
        cleaned_text = re.sub(r'https?:\/\/.*[\r\n]*', '', self.biography_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'来源：.*', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'作者：.*', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'编辑：.*', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'文/.*', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'征集.*', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'摄', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'[■※]', '', cleaned_text)
        # 去除重复文本
        cleaned_text = re.sub(r'\b(\S+)\b\s+\1\b', r'\1', cleaned_text, flags=re.MULTILINE)
        self.phases=cleaned_text.split()

    # 3.写文件
    def writefile(self):
        filename='../spider/No_'+str(self.count)+'.txt'
        with open(filename, "w", encoding="utf-8") as file:
            # 将列表中的每个元素写入文件，每个元素占一行
            for item in self.phases:
                file.write(item + "\n")
    
    # 4.读文件
    def readfile(self):
        filename='../spider/No_'+str(self.count)+'.txt'
        with open(filename, "r", encoding="utf-8") as fin: #F:\\Integrated_design\\test.txt
            self.text = fin.read()
        self.source.setText(self.text)
    
    # 5.分段与ner
    def NER(self):
        self.passages=self.text.split('\n')
        # print(self.passages)
        PERSONS=[]
        TIMES=[]
        LOCATIONS=[]
        for phase in self.passages:
            words=xmnlp.seg(phase)
            for word in words:
                ner=xmnlp.ner(word)
                for item in ner:
                    if item[1]=='PERSON':
                        PERSONS.append(item[0])
                    elif item[1]=='TIME':
                        TIMES.append(item[0])
                    elif item[1]=='LOCATION':
                        LOCATIONS.append(item[0])
        PERSON=[]
        TIME=[]
        LOCATION=[]
        persons=''
        times=''
        locations=''
        for person in PERSONS:
            if person not in PERSON:
                PERSON.append(person)   
        for time in TIMES:
            if time not in TIME:
                TIME.append(time)
        for location in LOCATIONS:
            if location not in LOCATION:
                LOCATION.append(location)

        for i in PERSON:
            persons+=i+'\n'
        for i in TIME:
            times+=i+'\n'
        for i in LOCATION:
            locations+=i+'\n'

        # print(PERSON)
        # print(TIME)
        # print(LOCATION)

        self.p.setText(persons)

        self.t.setText(times)
        self.l.setText(locations)


    
    # 6.摘要
    def textrank(self):
        results=''
        for phase in self.passages:
            sentence=wp.do(phase,1)
            results+="   "+sentence+'\n'
        self.result.setText(results)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MyWindow()
    # 展示窗口
    w.ui.show()
    sys.exit(app.exec_())