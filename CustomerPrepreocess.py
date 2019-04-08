# coding: utf-8
def process(df_cus,mode='predict',ProjectRouteCover_mode=None):
    import pandas as pd  
    import numpy as np
    from keras.utils import np_utils

    if(mode=='predict'):
        df_cus =df_cus[['CustomerCode','ProjectRouteCover','Gender','StudyTarget2','JobType','WatchTimes','LearningPurpose','ActionForce','CustomerExpRating','Age']].copy()
    elif(mode=='train'):
        df_cus = df_cus[['CustomerCode','IsOrder','ProjectRouteCover','KIStatus','Gender','StudyTarget2','JobType','WatchTimes','LearningPurpose','ActionForce','CustomerExpRating','Age']].copy()

    if(mode=='train'):
        # # 排除異常單 只取正常單
        df_cus = df_cus[df_cus.KIStatus == 1].copy()
        # # 億捷
        if (ProjectRouteCover_mode=='infojet'):
            df_cus=df_cus.loc[df_cus.ProjectRouteCover=='億捷',:]
        elif(ProjectRouteCover_mode=='other'):
            df_cus=df_cus.loc[df_cus.ProjectRouteCover!='億捷',:]

    # In[4]:


    df_cus.shape


    # In[5]:


    ProjectRouteCover_map={'FB':0,'其他':1,'廣告':2,'類關鍵字':3,'驚點聯播網':4,'億捷':5, '得易':6}
    Gender_map={'男':0, '女':1}
    StudyTarget2_map={'未知學習對象':0, '親子共學':1, '自學':2}
    JobType_map={'不提供':0,'沒工作':1,'有工作':2,'家管':3,'學生':4,'退休':5,'我忘了問':6,'問了客戶沒回答':0,'客戶忙沒問完':0}
    WatchTimes_map={0:0 ,1:1 ,2:2 ,3:0 ,4:0 ,5:0 ,9:3}
    LearningPurpose_map={'出國':1, '就是有興趣':2, '工作':3, '為了小孩':4, '無明確目的':5, '我忘了問':6, '問了客戶沒回答':0, '客戶忙沒問完':0}
    ActionForce_map={'上過補習班':0,
    '完全沒有':1,
    '買過書或教材':2,
    '有找過資訊未行動':3,
    '上過補習班;買過書或教材':4,
    '上過補習班;有找過資訊未行動':5,
    '有找過資訊未行動;完全沒有':6,
    '客戶忙沒問完':7,
    '上過補習班;完全沒有':8,
    '買過書或教材;完全沒有':9,
    '上過補習班;買過書或教材;有找過資訊未行動':10,
    '買過書或教材;有找過資訊未行動':11,
    '問了客戶沒回答':12,
    '上過補習班;買過書或教材;完全沒有':13}


    # In[6]:


    df_cus.ProjectRouteCover=df_cus.ProjectRouteCover.map(ProjectRouteCover_map)
    df_cus.Gender=df_cus.Gender.map(Gender_map)
    df_cus.StudyTarget2=df_cus.StudyTarget2.map(StudyTarget2_map)
    df_cus.JobType=df_cus.JobType.map(JobType_map)
    df_cus.WatchTimes=df_cus.WatchTimes.map(WatchTimes_map)
    df_cus.LearningPurpose=df_cus.LearningPurpose.map(LearningPurpose_map)
    df_cus.ActionForce=df_cus.ActionForce.map(ActionForce_map)


    # In[7]:


    #沒對應到的nan當其他=1
    df_cus.ProjectRouteCover=df_cus.ProjectRouteCover.fillna(1)
    #姓名補上未知 2
    df_cus.Gender=df_cus.Gender.fillna(2)
    #補上未知 0
    df_cus.StudyTarget2=df_cus.StudyTarget2.fillna(0)
    #補上未知 0
    df_cus.JobType=df_cus.JobType.fillna(0)
    #補上未知 0
    df_cus.WatchTimes=df_cus.WatchTimes.fillna(0)
    #補上未知 0
    df_cus.LearningPurpose=df_cus.LearningPurpose.fillna(0)
    #nan轉成未知14
    df_cus.ActionForce=df_cus.ActionForce.fillna(14)


    # # 類別數量

    # In[8]:


    df_cus.ProjectRouteCover.unique()


    # In[9]:


    df_cus.Gender.unique()


    # In[10]:


    df_cus.StudyTarget2.unique()


    # In[11]:


    df_cus.JobType.unique()


    # In[12]:


    df_cus.WatchTimes.unique()


    # In[13]:


    df_cus.LearningPurpose.unique()


    # In[14]:


    df_cus.ActionForce.unique()


    # # Age做one-hot

    # In[15]:


    def age_replace(age):
        if(np.isnan(age)):
            return age
        age=int(age/10)-1
        if age>7:
            age=7
        return age
    def age_trim(age):
        if(age<=10):
            return np.nan
        elif(age>85):
            return np.nan
        return age


    # In[16]:


    df_cus.Age.unique()


    # In[17]:


    #keep age>=10  age<=85
    df_cus.Age=df_cus.Age.apply(age_trim)


    # In[18]:


    df_cus.Age.unique()


    # In[19]:


    #量化到0~7
    df_cus.Age=df_cus.Age.apply(age_replace)


    # In[20]:


    Age_mode=2


    # In[21]:


    df_cus.Age.unique()


    # In[22]:


    #補眾數2
    df_cus.Age=df_cus.Age.fillna(2)


    # In[23]:


    df_cus.Age.unique()


    # # CustomerExpRating-數值資料

    # In[24]:

    if(mode=='train'):
        df_cus.IsOrder=df_cus.IsOrder.fillna(0)
        def fill_CustomerExpRating(v):
            if(v==1):
                return 5
            else:
                return 2.129

        #如果成交設5 否則補平均值2.129
        df_cus.loc[df_cus['CustomerExpRating'].isnull()==True,'CustomerExpRating']=df_cus.loc[df_cus['CustomerExpRating'].isnull()==True,'IsOrder'].apply(fill_CustomerExpRating)
    elif(mode=='predict'):
        df_cus.CustomerExpRating=df_cus.CustomerExpRating.fillna(3)

    # # 剩下如果有空drop

    # In[31]:


    df_cus=df_cus.dropna()
    df_cus=df_cus.reset_index(drop=True)


    # # one-hot

    # In[32]:


    df_cus=df_cus.loc[:,['CustomerCode','ProjectRouteCover','Gender','StudyTarget2','JobType','WatchTimes','LearningPurpose','ActionForce','Age','CustomerExpRating']]


    # In[33]:


    df_cus[:1]


    # In[34]:


    from keras.utils.np_utils import to_categorical
    ProjectRouteCover_one_hot=to_categorical(df_cus.ProjectRouteCover, 7)
    Gender_one_hot=to_categorical(df_cus.Gender, 3)
    StudyTarget2_hot=to_categorical(df_cus.StudyTarget2, 3)
    JobType_one_hot=to_categorical(df_cus.JobType, 7)
    WatchTimes_hot=to_categorical(df_cus.WatchTimes, 4)
    LearningPurpose_hot=to_categorical(df_cus.LearningPurpose, 7)
    ActionForce_hot=to_categorical(df_cus.ActionForce, 15)
    Age_hot=to_categorical(df_cus.Age, 9)


    # In[35]:


    ProjectRouteCover_one_hot= pd.DataFrame(ProjectRouteCover_one_hot)
    Gender_one_hot= pd.DataFrame(Gender_one_hot)
    StudyTarget2_hot= pd.DataFrame(StudyTarget2_hot)
    JobType_one_hot= pd.DataFrame(JobType_one_hot)
    WatchTimes_hot= pd.DataFrame(WatchTimes_hot)
    LearningPurpose_hot= pd.DataFrame(LearningPurpose_hot)
    ActionForce_hot= pd.DataFrame(ActionForce_hot)
    Age_hot= pd.DataFrame(Age_hot)


    # # normalization

    # In[36]:


    df_cus.CustomerExpRating=df_cus.CustomerExpRating/5
    df_cus.loc[df_cus.CustomerExpRating>1,['CustomerExpRating']]=1
    df_cus.loc[df_cus.CustomerExpRating<0,['CustomerExpRating']]=0


    # In[37]:


    df_CustomerCode=df_cus[['CustomerCode']].copy()
    df_CustomerExpRating=df_cus[['CustomerExpRating']].copy()


    # In[38]:


    len(df_CustomerCode)


    # In[39]:


    #concat
    con_list=[df_CustomerCode,df_CustomerExpRating,ProjectRouteCover_one_hot,Gender_one_hot,StudyTarget2_hot,JobType_one_hot,WatchTimes_hot,LearningPurpose_hot,ActionForce_hot,Age_hot]
    result=pd.concat(con_list, axis=1)
    col_name=['CustomerCode','CustomerExpRating']
    add_name=list(range(len(result.columns.tolist())-2))
    col_name.extend(add_name)
    result.columns=col_name


    # In[40]:


    result.shape


    

    return result
 

