import numpy as np




class hotPixels():

    def __init__(self,x_all=np.array([]),y_all=np.array([]),w_all=np.array([]),size_all=np.array([]),rebin=1):

        self.x=x_all
        self.y=y_all
        self.w=w_all
        self.size=size_all
        self.XBINS=2822
        self.YBINS=4144
        self.i_cut=[]
        self.j_cut=[]
   
        self.counts2d=np.array([])
        self.xedges=np.array([])
        self.yedges=np.array([])
        self.rebin=rebin
        
    def find_HotPixels(self, n_sigma=10,low_threshold=10, min_counts=10 ):
       myCut=np.where( (self.w>low_threshold))
       xbins2d=int(self.XBINS/self.rebin)
       ybins2d=int(self.YBINS/self.rebin)
       #if len(self.size!=0):
       #     myCut=np.where( (self.w>low_threshold)&(self.size==1) )
       #     print("cut=  (self.w>",low_threshold,")&(self.size==1) ")
       self.counts2d,  self.xedges, self.yedges= np.histogram2d(self.x[myCut],self.y[myCut],bins=[xbins2d, ybins2d ],range=[[0,self.XBINS],[0,self.YBINS]])
       self.counts2d=   self.counts2d.T 
      
       self.i_cut=[]
       self.j_cut=[]
       #for i in range(1, YBINS-1):
       for i in range(1, ybins2d-1):
             for j in  range(1, xbins2d-1):
        
                 counts=self.counts2d[i][j]
                 mysum2=0
                 for delta_i in range(-1,2):
                     for delta_j in range(-1,2):
                         mysum2+=self.counts2d[i+delta_i][j+delta_j]
           
                 #if counts<low_threshold:
                 if counts<min_counts:    
                     continue

                 
                 mysum2corr=(mysum2-counts)/8.
                 if (counts-mysum2corr)>n_sigma*np.sqrt(mysum2corr):   
                        print ("AAAAGGGHHHH noise!! couts=",counts," ave =",mysum2corr ," i=",i," j=",j)
                        self.i_cut.append(i)  #Y
                        self.j_cut.append(j)  #X

    def applyCuts(self):
        pixCut_all=None
        for i in range(0,len(self.j_cut)):
            print(i," (",100.*i/len(self.j_cut),"%) -- pix_x=",self.j_cut[i]," inf= ",self.xedges[self.j_cut[i]]," up=",self.xedges[self.j_cut[i]+1]  )
            print(i," -- pix_y=",self.i_cut[i]," inf= ",self.yedges[self.i_cut[i]]," up=",self.yedges[self.i_cut[i]+1]  )

            x_low=self.xedges[self.j_cut[i]]
            x_up=self.xedges[self.j_cut[i]+1]
            y_low=self.yedges[self.i_cut[i]]
            y_up=self.yedges[self.i_cut[i]+1]
    
            #pixCut=np.where(~(((self.x<=x_up)&(self.x>=x_low))&( (self.y<=y_up)&(self.y>=y_low) )))
            pixCut=(~(((self.x<=x_up)&(self.x>=x_low))&( (self.y<=y_up)&(self.y>=y_low) )))
            #print("pixCut=",pixCut)
            if i==0:
                pixCut_all=pixCut
            else:
                pixCut_all=(pixCut_all)&(pixCut)
          
           
        self.w=self.w[pixCut_all]
        self.x=self.x[pixCut_all]
        self.y=self.y[pixCut_all]
        if len(self.size!=0):
            self.size= self.size[pixCut_all]

                

    def get_cutVectors(self):         

       return  self.w,  self.x, self.y,   self.size

    def save_cuts(self,filename):
       np.savez(filename, j_cut=self.j_cut, i_cut=self.i_cut, xedges= self.xedges, yedges=self.yedges )

    def retrive_cuts(self,filename):
       data=np.load(filename)
       self.j_cut=data['j_cut']
       self.i_cut=data['i_cut']
       self.xedges=data['xedges']
       self.yedges=data['yedges']
       

   
    def do_all(self):
       print(" hotPixels: find hot pixels ")
       self.find_HotPixels()
       print(" hotPixels: apply cut ")
       self.applyCuts()

            
       return  self.w,  self.x, self.y,   self.size
   
