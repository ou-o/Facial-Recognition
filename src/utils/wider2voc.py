import os,cv2

rootdir="./data/wider_face" #widerface数据集所在目录

minsize2select=20 #widerface中有大量小人脸，只取20以上的来训练

if __name__ == "__main__":
    img_sets=["train","val"]
    for img_set in img_sets:

        imgdir=rootdir+"/WIDER_"+img_set+"/images"
        gtfilepath=rootdir+"/wider_face_split/wider_face_"+img_set+"_bbx_gt.txt"

        labelsdir=rootdir+"/WIDER_"+img_set+"/labels"
        if not os.path.exists(labelsdir):
            os.mkdir(labelsdir)

        img_f=open(rootdir+"/"+img_set+".txt","w")
        with open(gtfilepath,'r') as gtfile:
            while(True ):
                filename=gtfile.readline()[:-1]
                print (filename)
                if(filename==""):
                    break

                imgpath=imgdir+"/"+filename
                img=cv2.imread(imgpath)
                if not img.data:
                    break

                imgheight=img.shape[0]
                imgwidth=img.shape[1]
                filedir=labelsdir+"/"+filename.split('/')[0]
                if not os.path.exists(filedir):
                    os.mkdir(filedir)

                numbbox=int(gtfile.readline())
                bboxes=[]
                for i in range(numbbox):
                    line=gtfile.readline()
                    line=line.split()
                    line=line[0:4]
                    if(int(line[3])<=0 or int(line[2])<=0):
                        continue
                    x=int(line[0])
                    y=int(line[1])
                    width=int(line[2]) #边框宽度
                    height=int(line[3]) #边框高度
                    bbox=(x,y,width,height)
                    if width>=minsize2select and height>=minsize2select:
                        bboxes.append(bbox) #取得满足条件的方框

                if len(bboxes)==0:
                    continue #若没有脸，则不要这个图片
                txtpath=labelsdir+"/"+filename
                txtpath=txtpath[:-3]+"txt"
                ftxt=open(txtpath,'w')

                for i in range(len(bboxes)): #Annotation files
                    bbox=bboxes[i]
                    xcenter=(bbox[0]+bbox[2]*0.5)/imgwidth
                    ycenter=(bbox[1]+bbox[3]*0.5)/imgheight
                    wr=bbox[2]*1.0/imgwidth
                    hr=bbox[3]*1.0/imgheight
                    ftxt.write("0 "+str(xcenter)+" "+str(ycenter)+" "+str(wr)+" "+str(hr)+"\n")
                ftxt.close()
                #Define Train and Validation Sets
                img_f.write("data/wider_face/WIDER_"+img_set+"/images/"+filename+"\n")
        img_f.close()
