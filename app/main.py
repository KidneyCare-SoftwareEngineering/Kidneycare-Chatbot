#!/usr/bin/env python
# coding: utf-8
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

import nest_asyncio
from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

LINE_SECRET = os.getenv("LINE_SECRET")
LINE_TOKEN = os.getenv("LINE_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# หั่นคำใส่ LLM

# In[2]:


def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
    return documents

docs = load_documents("./")


# In[3]:


llm = OllamaLLM(
    model="scb10x/llama3.2-typhoon2-1b-instruct",
    temperature=0.3,
)

embeddings = OllamaEmbeddings(
    model="scb10x/llama3.2-typhoon2-1b-instruct"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

split_docs = text_splitter.split_documents(docs)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever()



# In[4]:


prompt_template = """
คุณคือผู้ให้คำปรึกษา ที่ชื่อว่า KidneyCare ผู้ช่วยให้คำปรึกษาที่มีความรู้ความเข้าใจลึกซึ้งในเรื่องการดูแลสุขภาพของผู้ป่วยโรคไตโดยเน้นไปทางด้านอาหารการกิน การคำนวณสารอาหาร คุณสามารถตอบคำถามเกี่ยวกับโรคไต การดูแลสุขภาพได้ 
ตอบคำถามต่อไปนี้โดยยึดตามข้อมูลที่ให้มาเท่านั้น และควรแน่ใจว่าคำตอบถูกต้องตามข้อมูลที่มีอยู่
ถ้าไม่มั่นใจว่ามีข้อมูลก็ควรบอกว่า "ขออภัยไม่สามารถตอบคำถามนี้ได้ คำถามของคุณจะถูกบันทึกไว้เพื่อปรับปรุงคุณภาพของเรา"
ใส่อีโมจิต่าง ๆ ในแต่ละคำตอบด้วย เพื่อความ Friendly มากขึ้น เช่น "✨", "👋"  
และตอบด้วยความชัดเจน กระชับ ในภาษาไทย ลงท้ายด้วยคำว่า "ครับ" เพื่อให้สุภาพ และแทนตัวเองด้วย "ผม" หากคำถามไหนเป็นคำถามง่าย ๆ สามารถตอบได้เลย ไม่ต้องให้คำตอบยาว ๆ แต่ถ้าผู้ใช้ต้องการรายละเอียด ให้ตอบรายละเอียดเพิ่มเติม แต่หากผู้ใช้ไม่บอกว่าต้องการรู้เยอะแค่ไหน ให้ถามก่อนเสมอ
โดยจะมีข้อมูลผู้ใช้ส่งไปในรูปแบบไฟล์ JSON มีชื่อตัวแปร โดย name คือชื่อ, gender คือ เพศ, weight คือน้ำหนักหน่วยกิโลกรัม, height คือส่วนสูงหน่วยเซนติเมตร, kidney_level คือระดับโรคไตของผู้ใช้งาน ห้ามตอบกลับเป็น JSON ที่ให้ไป

บริบท: {context}

คำถาม: {question}


คำตอบ:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# In[5]:


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def ask_question(data):
    result = qa_chain.invoke({"query": data})
    return result["result"]


# In[ ]:


question = "โรคไตผมต้องระวังอะไรบ้าง"

data = f"{question}"
answer = ask_question(data)
print(f"คำถาม: {question}")
print(f"คำตอบ: {answer}")


# FastAPI เรียกใช้กรณีรูป

# In[6]:


from fastapi import FastAPI, File, UploadFile
from dotenv import load_dotenv
import requests
import shutil


load_dotenv()
app = FastAPI()



ROBOFLOW_API_URL = f"https://classify.roboflow.com/foods-qfydk/2?api_key={os.getenv('ROBOFLOW_API_KEY')}"

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):

    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    with open(file_location, "rb") as image_file:
        response = requests.post(ROBOFLOW_API_URL, files={"file": image_file})

    return response.json()


# เรียก Line 

# In[ ]:


from dotenv import load_dotenv
from pyngrok import ngrok
import os
import json

load_dotenv()
app = FastAPI()

line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
ngrok.set_auth_token(os.getenv('NGROK_AUTH_TOKEN'))


# In[ ]:


import requests
import os
from linebot.models import MessageEvent, ImageMessage
from fastapi import FastAPI
import json

app = FastAPI()

def start_loading_animation(user_id):
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))}"
    }
    data = {"chatId": user_id}
    
    response = requests.post(url, headers=headers, json=data)



def loading_userInformation(user_id):
    rust_server_url = "https://backend-billowing-waterfall-4640.fly.dev/chatbot"
    try:
        response = requests.get(f"{rust_server_url}{user_id}")
        response.raise_for_status()  
        global user_information
        user_information = response.text
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
        return None 

def load_roboflow_mapping():
    with open("menuid.json", "r", encoding="utf-8") as file:
        # print("map อยู่")
        return json.load(file)


def get_name(roboflow_id, mapping):
    # print("getnaming")
    return mapping.get(str(roboflow_id)) 


@app.post("/callback")
async def callback(request: Request):
    signature = request.headers['X-Line-Signature']
    body = await request.body()

    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    return 'OK'


# ดึง ROBOFLOW
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_id = event.message.id

    user_id = event.source.user_id  
    start_loading_animation(user_id)  

    image_content = line_bot_api.get_message_content(message_id)

    image_path = f"temp_{message_id}.jpg"
    with open(image_path, "wb") as f:
        for chunk in image_content.iter_content():
            f.write(chunk)

    # ส่งไปยัง Roboflow 
    try:
        with open(image_path, "rb") as img:
            response = requests.post(ROBOFLOW_API_URL, files={"file": img})

        result = response.json()

        predict = result.get("predictions", [])
        if predict:
            global roboflow_id
            roboflow_id = predict[0].get("class")
            roboflow_mapping = load_roboflow_mapping()
            name = get_name(roboflow_id, roboflow_mapping)
        else :
            roboflow_id = None

        
    

        if roboflow_id != None : 
            line_bot_api.reply_message (
                event.reply_token,
                TextMessage(text=f"roboflow_id: {roboflow_id} ชื่อไทย: {name}")
            )
        else:
            line_bot_api.reply_message (
                event.reply_token,
                TextMessage(text="ขอโทษด้วยเราไม่สามารถตรวจจับรูปที่ไม่ใช่อาหารได้")
            )

    except Exception as e:
        print(f"err : {e}")

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)  

    




    
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id  
    # print(f"User ID: {user_id}")
    start_loading_animation(user_id)  
    loading_userInformation(user_id)
    

    

    # ดึง rust มาทำงาน
    text = event.message.text


    data = f"{text} {user_information}"
    if user_information == None :
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="ขออภัย Server เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง")
        )
    else :
        reply_text = ask_question(data)

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )


# ngrok_tunnel = ngrok.connect(8000)
# print('Webhook URL:', ngrok_tunnel.public_url + '/callback')

nest_asyncio.apply()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


